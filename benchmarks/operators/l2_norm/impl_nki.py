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


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    def _rsqrt(dst, var, row_size):
        """``dst <- 1 / sqrt(var)`` for an fp32 ``[row_size, 1]`` column.

        There is no direct reciprocal-square-root instruction (``nl.rsqrt`` is
        rejected as an ``nisa.activation`` op), and the plain
        ``sqrt`` + ``reciprocal`` composition uses hardware approximations with
        ~1e-5 relative error -- looser than the fp32 tolerance this operator is
        verified against. Two Newton-Raphson steps of
        ``r <- r * (1.5 - 0.5 * var * r^2)`` restore full fp32 accuracy; they run
        on ``[P, 1]`` vectors, so the cost is negligible.
        """
        std = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=std, op=nl.sqrt, data=var)
        nisa.reciprocal(dst=dst, data=std)

        half_var = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=half_var, data=var, op0=nl.multiply, operand0=0.5)

        correction = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        for _newton_step in range(2):
            nisa.tensor_tensor(dst=correction, data1=dst, data2=dst, op=nl.multiply)
            nisa.tensor_tensor(dst=correction, data1=correction, data2=half_var,
                               op=nl.multiply)
            # correction = 1.5 - 0.5 * var * r^2
            nisa.tensor_scalar(dst=correction, data=correction,
                               op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.5)
            nisa.tensor_tensor(dst=dst, data1=dst, data2=correction, op=nl.multiply)

    @nki.jit
    def l2_norm_kernel(a_input, eps, free_cap, block_size):
        """Row-wise L2 normalization over the last (free) dimension.

        Mirrors ``x * torch.rsqrt(x.square().sum(-1, keepdim=True) + eps)``:
        ``y[r, c] = x[r, c] / sqrt(sum_c x[r, c]^2 + eps)``.

        Args:
            a_input: input tensor [rows, n_cols] in HBM (fp32 / bf16 / fp16).
            eps: epsilon added to the squared norm (compile-time constant).

        Returns:
            [rows, n_cols] tensor in HBM with the same dtype as ``a_input``.

        Notes:
            * The reduction runs *along the free axis within each partition-row*,
              so it is a plain ``nisa.tensor_reduce`` on the Vector engine; the
              accumulator is always fp32 regardless of the input dtype.
            * Row blocks (and, on the wide path, column blocks) are clamped to the
              tensor extents with ``min(...)``, so every tile is allocated at its
              real extent and no load/store masking is required.
            * ``n_cols <= cap``: one pass, the whole row block lives in SBUF and is
              scaled in place.
            * ``n_cols > cap``: two passes over ``FALLBACK_TILE``-wide column
              blocks -- pass 1 accumulates ``sum(x^2)`` into an fp32 ``[P, 1]``
              accumulator, then ``rstd`` is computed once, then pass 2 re-loads
              each block and scales it.
        """
        kernel_assert(len(a_input.shape) == 2, "input must be 2D [rows, n_cols]")
        n_rows, n_cols = a_input.shape
        kernel_assert(min(n_cols, block_size) <= nl.tile_size.sbuf_fmax,
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

                sq_tile = nl.ndarray((row_size, n_cols), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sq_tile, op=nl.square, data=x_tile)

                sum_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=sum_sq, op=nl.add, data=sq_tile, axis=(1,))
                nisa.tensor_scalar(dst=sum_sq, data=sum_sq, op0=nl.add, operand0=eps)

                rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                _rsqrt(rstd, sum_sq, row_size)

                y_tile = nl.ndarray((row_size, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=rstd)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, 0:n_cols], src=y_tile)

            return out_hbm

        n_col_tiles = div_ceil(n_cols, block_size)
        for row_tile in range(n_row_tiles):
            row_start = row_tile * PMAX
            row_size = min(PMAX, n_rows - row_start)
            row_end = row_start + row_size

            # ---- pass 1: sum(x^2) per row (fp32 accumulator) ----------------
            sum_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=sum_sq, value=0.0)

            for col_tile in range(n_col_tiles):
                col_start = col_tile * block_size
                col_size = min(block_size, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                sq_tile = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sq_tile, op=nl.square, data=x_tile)

                part_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=part_sq, op=nl.add, data=sq_tile, axis=(1,))

                nisa.tensor_tensor(dst=sum_sq, data1=sum_sq, data2=part_sq, op=nl.add)

            nisa.tensor_scalar(dst=sum_sq, data=sum_sq, op0=nl.add, operand0=eps)

            rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            _rsqrt(rstd, sum_sq, row_size)

            # ---- pass 2: y = x * rstd ---------------------------------------
            for col_tile in range(n_col_tiles):
                col_start = col_tile * block_size
                col_size = min(block_size, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                y_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=y_tile, data=x_tile, op0=nl.multiply, operand0=rstd)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, col_start:col_end], src=y_tile)

        return out_hbm


_SEARCH_SPACE = [SimpleNamespace(free_cap=fc, block_size=bs) for fc, bs in ((8192, 2048), (2048, 2048), (2048, 512), (256, 512), (256, 256))]
_tuner = NkiAutotuner(l2_norm_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, eps: float = 1e-6, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("l2_norm NKI: int8 not supported")

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x_2d.shape), str(x_2d.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_2d, float(eps), cfg.free_cap, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = SimpleNamespace(free_cap=FP32_FREE_CAP if x_2d.dtype == torch.float32 else FREE_CAP, block_size=FALLBACK_TILE)
    out_2d = l2_norm_kernel(x_2d, float(eps), cfg.free_cap, cfg.block_size)
    return out_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
