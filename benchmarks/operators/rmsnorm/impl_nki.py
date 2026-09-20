from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# Hardware constants (NeuronCore-v2/v3): SBUF partition count and the maximum
# free-dimension size of an ``nc_matmul`` moving tile (== one fp32 PSUM bank).
PMAX = 128
MOVING_FMAX = 512

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
    def _rstd_newton(dst, mean_sq, row_size):
        """``dst <- 1 / sqrt(mean_sq)`` for an fp32 ``[row_size, 1]`` column.

        There is no direct reciprocal-square-root instruction (``nl.rsqrt`` is
        rejected as an ``nisa.activation`` op), and the plain
        ``sqrt`` + ``reciprocal`` composition uses hardware approximations with
        ~1e-5 relative error -- looser than the fp32 tolerance this operator is
        verified against. Two Newton-Raphson steps of
        ``r <- r * (1.5 - 0.5 * mean_sq * r^2)`` restore full fp32 accuracy;
        they run on ``[P, 1]`` vectors, so the cost is negligible.
        """
        std = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=std, op=nl.sqrt, data=mean_sq)
        nisa.reciprocal(dst=dst, data=std)

        half_mean_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=half_mean_sq, data=mean_sq, op0=nl.multiply, operand0=0.5)

        correction = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        for _newton_step in range(2):
            nisa.tensor_tensor(dst=correction, data1=dst, data2=dst, op=nl.multiply)
            nisa.tensor_tensor(dst=correction, data1=correction, data2=half_mean_sq,
                               op=nl.multiply)
            # correction = 1.5 - 0.5 * mean_sq * r^2
            nisa.tensor_scalar(dst=correction, data=correction,
                               op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.5)
            nisa.tensor_tensor(dst=dst, data1=dst, data2=correction, op=nl.multiply)

    def _broadcast_weight(dst, weight_row, ones_row, col_start, col_size):
        """Replicate ``weight_row[0, col_start:col_start+col_size]`` to all partitions.

        ``weight`` lives on a single SBUF partition but is needed on all 128 of
        them. An ``nc_matmul`` against a row of ones does the copy on the Tensor
        engine: ``ones[1, 128]^T @ row[1, C] -> [128, C]``. The moving operand's
        free dimension is capped at ``MOVING_FMAX``, so wide blocks are split.
        """
        for bcast_tile in range(div_ceil(col_size, MOVING_FMAX)):
            chunk_start = bcast_tile * MOVING_FMAX
            chunk_size = min(MOVING_FMAX, col_size - chunk_start)
            src_start = col_start + chunk_start
            src_end = src_start + chunk_size

            psum_weight = nl.ndarray((PMAX, chunk_size), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_weight, stationary=ones_row,
                           moving=weight_row[0:1, src_start:src_end])
            nisa.tensor_copy(dst=dst[0:PMAX, chunk_start:chunk_start + chunk_size],
                             src=psum_weight)

    @nki.jit
    def rmsnorm_kernel(a_input, weight_input, eps, free_cap, block_size):
        """Row-wise RMS normalization over the last (free) dimension.

        Mirrors ``F.rms_norm(x, (n_cols,), weight, eps)``:
        ``y[r, c] = x[r, c] / sqrt(mean_c x[r, c]^2 + eps) * weight[c]``.

        Args:
            a_input: input tensor [rows, n_cols] in HBM (fp32 / bf16 / fp16).
            weight_input: per-column scale [1, n_cols] in HBM.
            eps: epsilon added to the mean square (compile-time constant).

        Returns:
            [rows, n_cols] tensor in HBM with the same dtype as ``a_input``.

        Notes:
            * The reduction runs *along the free axis within each partition-row*,
              so it is a plain ``nisa.tensor_reduce`` on the Vector engine; the
              accumulator is always fp32 regardless of the input dtype.
            * ``mean(x^2)`` is a multiply by the compile-time constant
              ``1 / n_cols``, fused with the ``+ eps`` into one ``tensor_scalar``.
            * ``weight`` is broadcast from partition 0 to all 128 partitions with
              an ``nc_matmul`` against a row of ones.
            * Row blocks (and, on the wide path, column blocks) are clamped to the
              tensor extents with ``min(...)``, so every tile is allocated at its
              real extent and no load/store masking is required.
            * ``n_cols <= cap``: one pass, the whole row block lives in SBUF and is
              scaled in place.
            * ``n_cols > cap``: two passes over ``FALLBACK_TILE``-wide column
              blocks -- pass 1 accumulates ``sum(x^2)`` into an fp32 ``[P, 1]``
              accumulator, then ``rstd`` is computed once, then pass 2 re-loads
              each block and applies ``rstd * weight``.
        """
        kernel_assert(len(a_input.shape) == 2, "input must be 2D [rows, n_cols]")
        n_rows, n_cols = a_input.shape
        kernel_assert(len(weight_input.shape) == 2 and weight_input.shape[0] == 1
                      and weight_input.shape[1] == n_cols,
                      "weight must be [1, n_cols]")
        kernel_assert(min(n_cols, block_size) <= nl.tile_size.sbuf_fmax,
                      "column block exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((n_rows, n_cols), dtype=a_input.dtype, buffer=nl.shared_hbm)
        n_row_tiles = div_ceil(n_rows, PMAX)

        # ---- weight -> fp32, single partition -------------------------------
        weight_raw = nl.ndarray((1, n_cols), dtype=weight_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=weight_raw, src=weight_input[0:1, 0:n_cols])

        weight_row = nl.ndarray((1, n_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=weight_row, src=weight_raw)

        ones_row = nl.ndarray((1, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)

        # mean(x^2) == sum(x^2) * (1 / n_cols); ``n_cols`` is a compile-time int.
        inv_n_cols = 1.0 / float(n_cols)


        if n_cols <= free_cap:
            # The whole row fits in SBUF, so the broadcast weight row is hoisted
            # out of the row loop and reused by every row block.
            weight_bcast = nl.ndarray((PMAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
            _broadcast_weight(weight_bcast, weight_row, ones_row, 0, n_cols)

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

                mean_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=mean_sq, op=nl.add, data=sq_tile, axis=(1,))
                nisa.tensor_scalar(dst=mean_sq, data=mean_sq,
                                   op0=nl.multiply, operand0=inv_n_cols,
                                   op1=nl.add, operand1=eps)

                rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                _rstd_newton(rstd, mean_sq, row_size)

                normalized = nl.ndarray((row_size, n_cols), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=normalized, data=x_tile,
                                   op0=nl.multiply, operand0=rstd)

                y_tile = nl.ndarray((row_size, n_cols), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=y_tile, data1=normalized,
                                   data2=weight_bcast[0:row_size, 0:n_cols],
                                   op=nl.multiply)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, 0:n_cols], src=y_tile)

            return out_hbm

        n_col_tiles = div_ceil(n_cols, block_size)
        # A full [128, n_cols] fp32 broadcast would not fit in SBUF on this path,
        # so the weight block is broadcast inside the second pass instead.
        weight_bcast = nl.ndarray((PMAX, block_size), dtype=nl.float32, buffer=nl.sbuf)

        for row_tile in range(n_row_tiles):
            row_start = row_tile * PMAX
            row_size = min(PMAX, n_rows - row_start)
            row_end = row_start + row_size

            # ---- pass 1: sum(x^2) per row (fp32 accumulator) ----------------
            mean_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=mean_sq, value=0.0)

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

                nisa.tensor_tensor(dst=mean_sq, data1=mean_sq, data2=part_sq, op=nl.add)

            nisa.tensor_scalar(dst=mean_sq, data=mean_sq,
                               op0=nl.multiply, operand0=inv_n_cols,
                               op1=nl.add, operand1=eps)

            rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            _rstd_newton(rstd, mean_sq, row_size)

            # ---- pass 2: y = x * rstd * weight ------------------------------
            for col_tile in range(n_col_tiles):
                col_start = col_tile * block_size
                col_size = min(block_size, n_cols - col_start)
                col_end = col_start + col_size

                _broadcast_weight(weight_bcast, weight_row, ones_row, col_start, col_size)

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                normalized = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=normalized, data=x_tile,
                                   op0=nl.multiply, operand0=rstd)

                y_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=y_tile, data1=normalized,
                                   data2=weight_bcast[0:row_size, 0:col_size],
                                   op=nl.multiply)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, col_start:col_end], src=y_tile)

        return out_hbm


_SEARCH_SPACE = [SimpleNamespace(free_cap=fc, block_size=bs) for fc, bs in ((8192, 2048), (2048, 2048), (2048, 512), (256, 512), (256, 256))]
_tuner = NkiAutotuner(rmsnorm_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("rmsnorm NKI: int8 not supported")

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    weight_2d = weight.reshape(1, -1)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x_2d.shape), str(x_2d.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_2d, weight_2d, float(eps), cfg.free_cap, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = SimpleNamespace(free_cap=FP32_FREE_CAP if x_2d.dtype == torch.float32 else FREE_CAP, block_size=FALLBACK_TILE)
    out_2d = rmsnorm_kernel(x_2d, weight_2d, float(eps), cfg.free_cap, cfg.block_size)
    return out_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
