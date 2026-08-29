from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def weight_dequant_kernel(X, S, TILE_SIZE, block_size):
        """Block-wise weight dequantization: ``out[m, n] = X[m, n] * S[m // T, n // T]``.

        ``S`` holds one scale per ``TILE_SIZE x TILE_SIZE`` block of ``X``.

        Args:
            X: [M, N] quantized weights in HBM.
            S: [ceil(M/T), ceil(N/T)] per-block scales in HBM (same dtype as X).
            TILE_SIZE: block edge length T, a multiple of PMAX (checked in ``run``).

        Returns:
            [M, N] tensor in HBM with the same dtype as ``X``.

        Notes:
            * Because ``TILE_SIZE`` is a multiple of ``PMAX``, a ``PMAX``-row tile never
              straddles two scale rows, so one scale row of ``S`` serves every row tile
              inside a ``TILE_SIZE``-row band.
            * The scalar broadcast is done in two cheap steps rather than by
              materializing a full [PMAX, TILE_SIZE] scale tile:
              1. one DMA per scale row with a partition stride of 0
                 (``S.ap(pattern=[[0, PMAX], [1, s_cols]])``) replicates the whole
                 scale row to all 128 partitions in a single instruction;
              2. ``nisa.tensor_scalar`` with ``operand0`` = the [PMAX, 1] column for
                 that block broadcasts along the free axis in hardware.
              ``operand0`` must be float32 (the MLIR verifier rejects a half-precision
              ``operand0``), hence the fp32 copy of the broadcast row.
            * Boundary tiles are clamped (tile sized to the surviving extent) instead of
              masked, so no out-of-range element is ever loaded, multiplied or stored.
        """
        kernel_assert(len(X.shape) == 2, "X must be 2D [M, N]")
        kernel_assert(len(S.shape) == 2, "S must be 2D [M/T, N/T]")

        M, N = X.shape
        s_rows = div_ceil(M, TILE_SIZE)
        s_cols = div_ceil(N, TILE_SIZE)
        kernel_assert(S.shape[0] >= s_rows and S.shape[1] >= s_cols,
                      "S is too small for the requested block grid")

        hbm_result = nl.ndarray((M, N), dtype=X.dtype, buffer=nl.shared_hbm)

        s_stride = S.shape[1]

        for sr in range(s_rows):
            band_start = sr * TILE_SIZE
            band_size = min(TILE_SIZE, M - band_start)

            # Replicate scale row ``sr`` to every partition: partition stride 0.
            s_row_bcast = nl.ndarray((PMAX, s_cols), dtype=S.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=s_row_bcast,
                src=S.ap(pattern=[[0, PMAX], [1, s_cols]], offset=sr * s_stride),
            )

            s_row_f32 = nl.ndarray((PMAX, s_cols), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s_row_f32, src=s_row_bcast)

            for rt in range(div_ceil(band_size, block_size)):
                row_start = band_start + rt * block_size
                row_size = min(block_size, M - row_start)

                for sc in range(s_cols):
                    col_start = sc * TILE_SIZE
                    col_size = min(TILE_SIZE, N - col_start)

                    x_tile = nl.ndarray((row_size, col_size), dtype=X.dtype,
                                        buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=x_tile,
                        src=X[row_start:row_start + row_size,
                              col_start:col_start + col_size],
                    )

                    out_tile = nl.ndarray((row_size, col_size), dtype=X.dtype,
                                          buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=out_tile, data=x_tile, op0=nl.multiply,
                                       operand0=s_row_f32[0:row_size, sc:sc + 1])

                    nisa.dma_copy(
                        dst=hbm_result[row_start:row_start + row_size,
                                       col_start:col_start + col_size],
                        src=out_tile,
                    )

        return hbm_result


_DEFAULT_CONFIG = SimpleNamespace(block_size=128)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (32, 64, 128)]
_tuner = NkiAutotuner(weight_dequant_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if TILE_SIZE % PMAX != 0:
        raise NotImplementedError(
            f"weight_dequant NKI: TILE_SIZE ({TILE_SIZE}) must be a multiple of {PMAX}"
        )
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(X.shape), str(X.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (X, S, TILE_SIZE, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    return weight_dequant_kernel(X, S, TILE_SIZE, cfg.block_size)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
