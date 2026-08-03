import torch

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# Hardware constants (NeuronCore-v2/v3): SBUF partition count and the maximum
# free-dimension size of an ``nc_matmul`` moving tile (== one fp32 PSUM bank).
P_MAX = 128
MOVING_FMAX = 512


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def batch_norm_kernel(x_hbm, gamma_hbm, beta_hbm, eps):
        """Batch normalization with training-mode (batch) statistics.

        Mirrors ``F.batch_norm(x, None, None, gamma, beta, training=True, eps=eps)``:
        per-channel mean/variance are computed over the batch dimension N
        (biased variance, i.e. divided by N), then
        ``y[n, c] = (x[n, c] - mean[c]) * rstd[c] * gamma[c] + beta[c]``.

        Implemented as ``y = x * scale + shift`` with
        ``scale = gamma / sqrt(var + eps)`` and ``shift = beta - mean * scale``.

        Args:
            x_hbm: input tensor [N, C] in HBM (fp32 / bf16 / fp16).
            gamma_hbm: per-channel scale [1, C] in HBM.
            beta_hbm: per-channel shift [1, C] in HBM.
            eps: variance epsilon (compile-time constant).

        Returns:
            [N, C] tensor in HBM with the same dtype as ``x_hbm``.

        Notes:
            * Reduction over N is a cross-partition reduction. It is done in two
              stages: an fp32 element-wise accumulation into a [128, C] SBUF tile
              (Vector engine), followed by a single ``nc_matmul`` against a column
              of ones which folds the 128 partitions into one.
            * N == 1 degenerates to ``y = beta`` (batch variance is zero); this is
              obtained for free by setting ``scale = 0`` and ``shift = beta``.
        """
        N, C = x_hbm.shape
        kernel_assert(len(x_hbm.shape) == 2, "input must be 2D [N, C]")
        kernel_assert(C <= nl.tile_size.sbuf_fmax, "C exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((N, C), dtype=x_hbm.dtype, buffer=nl.shared_hbm)

        n_row_tiles = div_ceil(N, P_MAX)
        n_col_tiles = div_ceil(C, MOVING_FMAX)

        # ---- gamma / beta -> fp32, single partition -------------------------
        gamma_raw = nl.ndarray((1, C), dtype=gamma_hbm.dtype, buffer=nl.sbuf)
        beta_raw = nl.ndarray((1, C), dtype=beta_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=gamma_raw, src=gamma_hbm[0:1, 0:C])
        nisa.dma_copy(dst=beta_raw, src=beta_hbm[0:1, 0:C])

        gamma_f32 = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        beta_f32 = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gamma_f32, src=gamma_raw)
        nisa.tensor_copy(dst=beta_f32, src=beta_raw)

        scale_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
        shift_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)

        if N == 1:
            # Batch variance is undefined/zero with a single sample: y = beta.
            nisa.memset(dst=scale_row, value=0.0)
            nisa.tensor_copy(dst=shift_row, src=beta_f32)
        else:
            # ---- pass 1a: partial sum / sum-of-squares over N, kept per partition
            acc_sum = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.sbuf)
            acc_sq = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc_sum, value=0.0)
            nisa.memset(dst=acc_sq, value=0.0)

            for row_tile in nl.affine_range(n_row_tiles):
                row_start = row_tile * P_MAX
                row_size = min(P_MAX, N - row_start)

                x_tile = nl.ndarray((row_size, C), dtype=x_hbm.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=x_hbm[row_start:row_start + row_size, 0:C])

                # Squaring runs on the Scalar (activation) engine so that it
                # overlaps with the Vector-engine accumulations below.
                sq_tile = nl.ndarray((row_size, C), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sq_tile, op=nl.square, data=x_tile)

                nisa.tensor_tensor(dst=acc_sum[0:row_size, 0:C],
                                   data1=acc_sum[0:row_size, 0:C], data2=x_tile, op=nl.add)
                nisa.tensor_tensor(dst=acc_sq[0:row_size, 0:C],
                                   data1=acc_sq[0:row_size, 0:C], data2=sq_tile, op=nl.add)

            # ---- pass 1b: fold the 128 partitions into one (ones^T @ acc) ----
            ones_col = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=ones_col, value=1.0)

            total_sum = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            total_sq = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)

            for col_tile in nl.affine_range(n_col_tiles):
                col_start = col_tile * MOVING_FMAX
                col_size = min(MOVING_FMAX, C - col_start)
                col_end = col_start + col_size

                psum_sum = nl.ndarray((1, col_size), dtype=nl.float32, buffer=nl.psum)
                psum_sq = nl.ndarray((1, col_size), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=psum_sum, stationary=ones_col,
                               moving=acc_sum[0:P_MAX, col_start:col_end])
                nisa.nc_matmul(dst=psum_sq, stationary=ones_col,
                               moving=acc_sq[0:P_MAX, col_start:col_end])
                nisa.tensor_copy(dst=total_sum[0:1, col_start:col_end], src=psum_sum)
                nisa.tensor_copy(dst=total_sq[0:1, col_start:col_end], src=psum_sq)

            # ---- statistics: mean, var, scale, shift (single partition) -----
            mean_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            mean_sq_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mean_row, data=total_sum,
                               op0=nl.multiply, operand0=1.0 / N)
            nisa.tensor_scalar(dst=mean_sq_row, data=total_sq,
                               op0=nl.multiply, operand0=1.0 / N)

            var_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=var_row, data1=mean_row, data2=mean_row, op=nl.multiply)
            nisa.tensor_tensor(dst=var_row, data1=mean_sq_row, data2=var_row, op=nl.subtract)
            # var = max(var, 0) + eps  (guard against tiny negative round-off)
            nisa.tensor_scalar(dst=var_row, data=var_row,
                               op0=nl.maximum, operand0=0.0, op1=nl.add, operand1=eps)

            std_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            rstd_row = nl.ndarray((1, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=std_row, op=nl.sqrt, data=var_row)
            nisa.reciprocal(dst=rstd_row, data=std_row)

            nisa.tensor_tensor(dst=scale_row, data1=rstd_row, data2=gamma_f32, op=nl.multiply)
            nisa.tensor_tensor(dst=shift_row, data1=mean_row, data2=scale_row, op=nl.multiply)
            nisa.tensor_tensor(dst=shift_row, data1=beta_f32, data2=shift_row, op=nl.subtract)

        # ---- broadcast scale / shift from partition 0 to all 128 partitions --
        # ones[1, 128]^T @ row[1, col_size] -> [128, col_size]
        ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)

        scale_bcast = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.sbuf)
        shift_bcast = nl.ndarray((P_MAX, C), dtype=nl.float32, buffer=nl.sbuf)

        for col_tile in nl.affine_range(n_col_tiles):
            col_start = col_tile * MOVING_FMAX
            col_size = min(MOVING_FMAX, C - col_start)
            col_end = col_start + col_size

            psum_scale = nl.ndarray((P_MAX, col_size), dtype=nl.float32, buffer=nl.psum)
            psum_shift = nl.ndarray((P_MAX, col_size), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_scale, stationary=ones_row,
                           moving=scale_row[0:1, col_start:col_end])
            nisa.nc_matmul(dst=psum_shift, stationary=ones_row,
                           moving=shift_row[0:1, col_start:col_end])
            nisa.tensor_copy(dst=scale_bcast[0:P_MAX, col_start:col_end], src=psum_scale)
            nisa.tensor_copy(dst=shift_bcast[0:P_MAX, col_start:col_end], src=psum_shift)

        # ---- pass 2: y = x * scale + shift ----------------------------------
        for row_tile in nl.affine_range(n_row_tiles):
            row_start = row_tile * P_MAX
            row_size = min(P_MAX, N - row_start)

            x_tile = nl.ndarray((row_size, C), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x_hbm[row_start:row_start + row_size, 0:C])

            scaled = nl.ndarray((row_size, C), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=scaled, data1=x_tile,
                               data2=scale_bcast[0:row_size, 0:C], op=nl.multiply)

            y_tile = nl.ndarray((row_size, C), dtype=x_hbm.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=y_tile, data1=scaled,
                               data2=shift_bcast[0:row_size, 0:C], op=nl.add)

            nisa.dma_copy(dst=out_hbm[row_start:row_start + row_size, 0:C], src=y_tile)

        return out_hbm


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if input.dtype == torch.int8:
        raise NotImplementedError("batch_normalization NKI: int8 not supported")
    gamma_2d = gamma.reshape(1, -1)
    beta_2d = beta.reshape(1, -1)
    return batch_norm_kernel(input, gamma_2d, beta_2d, float(eps))


def get_last_config() -> dict | None:
    return None
