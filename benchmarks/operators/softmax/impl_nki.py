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

# Rows whose full free extent fits in a single SBUF tile take the one-pass path;
# wider rows are streamed in ``FALLBACK_TILE``-wide column blocks. fp32 rows hold
# twice the bytes per element, so they switch to the streaming path earlier.
FREE_CAP = 8192
FP32_FREE_CAP = 2048
FALLBACK_TILE = 2048

# Seed for the running row-max accumulator. Smaller than -max(float32), so any
# real fp32/bf16/fp16 element wins the first ``nl.maximum`` fold.
NEG_INF = -3.0e38


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def softmax_online_kernel(a_input, free_cap, fallback_tile):
        """Row-wise softmax over the last (free) dimension.

        Mirrors ``torch.softmax(x, dim=-1)`` in its numerically stable form:
        ``y[r, c] = exp(x[r, c] - max_c x[r, c]) / sum_c exp(x[r, c] - max_c x[r, c])``.

        Args:
            a_input: input tensor [rows, n_cols] in HBM (fp32 / bf16 / fp16).

        Returns:
            [rows, n_cols] tensor in HBM with the same dtype as ``a_input``.

        Notes:
            * Both reductions run *along the free axis within each partition-row*,
              so they are plain ``nisa.tensor_reduce`` ops on the Vector engine;
              max and sum accumulators are always fp32 regardless of input dtype.
            * Row blocks (and, on the wide path, column blocks) are clamped to the
              tensor extents with ``min(...)``, so every tile is allocated at its
              real extent and no load/store masking is required -- in particular
              no ``-inf`` padding is ever fed to the max reduction.
            * ``n_cols <= cap``: one pass, the whole row block lives in SBUF and is
              max-shifted, exponentiated and normalized in place.
            * ``n_cols > cap``: three passes over ``FALLBACK_TILE``-wide column
              blocks -- pass 1 folds each block's max into a running fp32 ``[P, 1]``
              row max, pass 2 re-loads each block and accumulates
              ``sum(exp(x - row_max))``, pass 3 re-loads each block once more and
              scales it by ``1 / row_sum``. The running max is a true max over all
              blocks (not the last block's), which is what makes passes 2 and 3
              consistent with each other.
        """
        kernel_assert(len(a_input.shape) == 2, "input must be 2D [rows, n_cols]")
        n_rows, n_cols = a_input.shape
        kernel_assert(min(n_cols, fallback_tile) <= nl.tile_size.sbuf_fmax,
                      "column block exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((n_rows, n_cols), dtype=a_input.dtype, buffer=nl.shared_hbm)
        n_row_tiles = div_ceil(n_rows, PMAX)

        if n_cols <= free_cap:
            for row_tile in range(n_row_tiles):
                row_start = row_tile * PMAX
                row_size = min(PMAX, n_rows - row_start)
                row_end = row_start + row_size

                # DMA cannot convert dtypes: load in the input dtype and let the
                # Scalar/Vector engines widen to fp32 on their way out.
                x_tile = nl.ndarray((row_size, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, 0:n_cols])

                row_max = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=row_max, op=nl.maximum, data=x_tile, axis=(1,))

                # work <- exp(x - row_max); the [P, 1] operand broadcasts along free.
                work = nl.ndarray((row_size, n_cols), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=work, data=x_tile, op0=nl.subtract, operand0=row_max)
                nisa.activation(dst=work, op=nl.exp, data=work)

                row_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=row_sum, op=nl.add, data=work, axis=(1,))

                # ``nl.divide`` is not a valid ISA operator; reciprocal + multiply.
                inv_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.reciprocal(dst=inv_sum, data=row_sum)

                y_tile = nl.ndarray((row_size, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=work, op0=nl.multiply, operand0=inv_sum)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, 0:n_cols], src=y_tile)

            return out_hbm

        n_col_tiles = div_ceil(n_cols, fallback_tile)
        for row_tile in range(n_row_tiles):
            row_start = row_tile * PMAX
            row_size = min(PMAX, n_rows - row_start)
            row_end = row_start + row_size

            # ---- pass 1: running max(x) per row (fp32 accumulator) -----------
            row_max = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=row_max, value=NEG_INF)

            for col_tile in range(n_col_tiles):
                col_start = col_tile * fallback_tile
                col_size = min(fallback_tile, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                block_max = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=block_max, op=nl.maximum, data=x_tile, axis=(1,))

                nisa.tensor_tensor(dst=row_max, data1=row_max, data2=block_max, op=nl.maximum)

            # ---- pass 2: sum(exp(x - row_max)) per row -----------------------
            row_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=row_sum, value=0.0)

            for col_tile in range(n_col_tiles):
                col_start = col_tile * fallback_tile
                col_size = min(fallback_tile, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                work = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=work, data=x_tile, op0=nl.subtract, operand0=row_max)
                nisa.activation(dst=work, op=nl.exp, data=work)

                part_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=part_sum, op=nl.add, data=work, axis=(1,))

                nisa.tensor_tensor(dst=row_sum, data1=row_sum, data2=part_sum, op=nl.add)

            inv_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_sum, data=row_sum)

            # ---- pass 3: y = exp(x - row_max) / row_sum ----------------------
            for col_tile in range(n_col_tiles):
                col_start = col_tile * fallback_tile
                col_size = min(fallback_tile, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                work = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=work, data=x_tile, op0=nl.subtract, operand0=row_max)
                nisa.activation(dst=work, op=nl.exp, data=work)

                y_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=work, op0=nl.multiply, operand0=inv_sum)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, col_start:col_end], src=y_tile)

        return out_hbm


_SEARCH_SPACE = [SimpleNamespace(free_cap=fc, fallback_tile=ft)
                 for fc, ft in ((8192, 2048), (2048, 2048), (2048, 512),
                                (256, 512), (256, 256))]
_tuner = NkiAutotuner(softmax_online_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x.shape), str(x.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x, cfg.free_cap, cfg.fallback_tile),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = SimpleNamespace(
            free_cap=FP32_FREE_CAP if x.dtype == torch.float32 else FREE_CAP,
            fallback_tile=FALLBACK_TILE,
        )
    return softmax_online_kernel(x, cfg.free_cap, cfg.fallback_tile)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
