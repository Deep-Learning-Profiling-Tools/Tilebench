from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# Hardware constant (NeuronCore-v2/v3): number of SBUF partitions.
PMAX = 128


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def mean_rowwise_kernel(a_input, block_size_m):
        """Row-wise mean over the last (free) dimension.

        Mirrors ``x.mean(dim=1, dtype=torch.float32)``:
        ``y[r] = (sum_c x[r, c]) / n_cols``.

        Args:
            a_input: input tensor [n_rows, n_cols] in HBM (fp32 / bf16 / fp16).

        Returns:
            [n_rows, 1] fp32 tensor in HBM. The result is always fp32 -- the
            reference accumulates and returns in fp32 regardless of the input
            dtype, so narrowing the output here would lose that precision.

        Notes:
            * The reduction runs *along the free axis within each partition-row*,
              so it is a plain ``nisa.tensor_reduce`` on the Vector engine. The
              engine always accumulates in fp32 internally, so the input tile is
              loaded in its native dtype (DMA cannot convert dtypes anyway) and
              widened on its way into the fp32 accumulator.
            * Row blocks are clamped to the tensor extent with ``min(...)``, so
              every tile is allocated at its real extent and no load/store
              masking is required for a trailing partial block.
            * The whole row block (all ``n_cols`` columns) lives in SBUF in one
              tile; there is no column tiling.
            * The division by ``n_cols`` is folded into a ``tensor_scalar``
              multiply by the reciprocal, computed once in Python at trace time
              (there is no ISA divide op).
        """
        kernel_assert(len(a_input.shape) == 2, "input must be 2D [n_rows, n_cols]")
        n_rows, n_cols = a_input.shape
        kernel_assert(n_cols <= nl.tile_size.sbuf_fmax,
                      "row width exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((n_rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        n_row_tiles = div_ceil(n_rows, block_size_m)
        inv_n_cols = 1.0 / n_cols

        for row_tile in range(n_row_tiles):
            row_start = row_tile * block_size_m
            row_size = min(block_size_m, n_rows - row_start)
            row_end = row_start + row_size

            x_tile = nl.ndarray((row_size, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, 0:n_cols])

            row_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=row_sum, op=nl.add, data=x_tile, axis=(1,),
                               keepdims=True)

            row_mean = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=row_mean, data=row_sum,
                               op0=nl.multiply, operand0=inv_n_cols)

            nisa.dma_copy(dst=out_hbm[row_start:row_end, 0:1], src=row_mean)

        return out_hbm


_DEFAULT_CONFIG = SimpleNamespace(block_size_m=128)
_SEARCH_SPACE = [SimpleNamespace(block_size_m=b) for b in (32, 64, 128)]
_tuner = NkiAutotuner(mean_rowwise_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("mean reduction NKI: int8 not supported")

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x, cfg.block_size_m),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = mean_rowwise_kernel(x, cfg.block_size_m)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
