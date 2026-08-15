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

# Free-dimension (column) block size used when streaming a row block.
FREE_TILE = 2048


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


if nki is not None:
    @nki.jit
    def layernorm_kernel(a_input, weight_input, bias_input, eps):
        """Row-wise layer normalization over the last (free) dimension.

        Mirrors ``F.layer_norm(x, (n_cols,), weight, bias, eps)``:
        per-row mean/variance are computed over the ``n_cols`` columns (biased
        variance, i.e. divided by ``n_cols``), then
        ``y[r, c] = (x[r, c] - mean[r]) * rstd[r] * weight[c] + bias[c]``.

        Args:
            a_input: input tensor [rows, n_cols] in HBM (fp32 / bf16 / fp16).
            weight_input: per-column scale [1, n_cols] in HBM.
            bias_input: per-column shift [1, n_cols] in HBM.
            eps: variance epsilon (compile-time constant).

        Returns:
            [rows, n_cols] tensor in HBM with the same dtype as ``a_input``.

        Notes:
            * The statistics reduction runs *along the free axis within each
              partition-row*, so it is a plain ``nisa.tensor_reduce`` on the
              Vector engine; partial results from each column block are
              accumulated into fp32 [P, 1] accumulators.
            * ``weight``/``bias`` live on a single partition but are needed on
              all 128 partitions. They are broadcast once with an ``nc_matmul``
              against a row of ones (``ones[1, 128]^T @ row[1, C] -> [128, C]``).
            * Row blocks and column blocks are clamped to the tensor extents
              (``min(...)``) so no masking / zero-padding is required.
        """
        kernel_assert(len(a_input.shape) == 2, "input must be 2D [rows, n_cols]")
        n_rows, n_cols = a_input.shape
        kernel_assert(n_cols <= nl.tile_size.sbuf_fmax, "n_cols exceeds the SBUF free dimension")

        out_hbm = nl.ndarray((n_rows, n_cols), dtype=a_input.dtype, buffer=nl.shared_hbm)

        n_row_tiles = div_ceil(n_rows, P_MAX)
        n_col_tiles = div_ceil(n_cols, FREE_TILE)
        n_bcast_tiles = div_ceil(n_cols, MOVING_FMAX)

        # ---- weight / bias -> fp32, single partition ------------------------
        weight_raw = nl.ndarray((1, n_cols), dtype=weight_input.dtype, buffer=nl.sbuf)
        bias_raw = nl.ndarray((1, n_cols), dtype=bias_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=weight_raw, src=weight_input[0:1, 0:n_cols])
        nisa.dma_copy(dst=bias_raw, src=bias_input[0:1, 0:n_cols])

        weight_row = nl.ndarray((1, n_cols), dtype=nl.float32, buffer=nl.sbuf)
        bias_row = nl.ndarray((1, n_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=weight_row, src=weight_raw)
        nisa.tensor_copy(dst=bias_row, src=bias_raw)

        # ---- broadcast weight / bias from partition 0 to all 128 partitions --
        ones_row = nl.ndarray((1, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)

        weight_bcast = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)
        bias_bcast = nl.ndarray((P_MAX, n_cols), dtype=nl.float32, buffer=nl.sbuf)

        for bcast_tile in nl.affine_range(n_bcast_tiles):
            col_start = bcast_tile * MOVING_FMAX
            col_size = min(MOVING_FMAX, n_cols - col_start)
            col_end = col_start + col_size

            psum_weight = nl.ndarray((P_MAX, col_size), dtype=nl.float32, buffer=nl.psum)
            psum_bias = nl.ndarray((P_MAX, col_size), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=psum_weight, stationary=ones_row,
                           moving=weight_row[0:1, col_start:col_end])
            nisa.nc_matmul(dst=psum_bias, stationary=ones_row,
                           moving=bias_row[0:1, col_start:col_end])
            nisa.tensor_copy(dst=weight_bcast[0:P_MAX, col_start:col_end], src=psum_weight)
            nisa.tensor_copy(dst=bias_bcast[0:P_MAX, col_start:col_end], src=psum_bias)

        inv_n_cols = 1.0 / float(n_cols)

        for row_tile in nl.affine_range(n_row_tiles):
            row_start = row_tile * P_MAX
            row_size = min(P_MAX, n_rows - row_start)
            row_end = row_start + row_size

            # ---- pass 1: sum(x) and sum(x^2) per row (fp32 accumulators) ----
            sum_x = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            sum_x2 = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=sum_x, value=0.0)
            nisa.memset(dst=sum_x2, value=0.0)

            for col_tile in nl.affine_range(n_col_tiles):
                col_start = col_tile * FREE_TILE
                col_size = min(FREE_TILE, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                # Squaring runs on the Scalar (activation) engine so that it
                # overlaps with the Vector-engine reductions below.
                sq_tile = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=sq_tile, op=nl.square, data=x_tile)

                part_sum = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                part_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=part_sum, op=nl.add, data=x_tile, axis=(1,))
                nisa.tensor_reduce(dst=part_sq, op=nl.add, data=sq_tile, axis=(1,))

                nisa.tensor_tensor(dst=sum_x, data1=sum_x, data2=part_sum, op=nl.add)
                nisa.tensor_tensor(dst=sum_x2, data1=sum_x2, data2=part_sq, op=nl.add)

            # ---- statistics: mean, var, rstd (per-partition [P, 1] vectors) --
            mean = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            mean_sq = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mean, data=sum_x, op0=nl.multiply, operand0=inv_n_cols)
            nisa.tensor_scalar(dst=mean_sq, data=sum_x2, op0=nl.multiply, operand0=inv_n_cols)

            var = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=var, data1=mean, data2=mean, op=nl.multiply)
            nisa.tensor_tensor(dst=var, data1=mean_sq, data2=var, op=nl.subtract)
            # var = max(var, 0) + eps  (guard against tiny negative round-off)
            nisa.tensor_scalar(dst=var, data=var,
                               op0=nl.maximum, operand0=0.0, op1=nl.add, operand1=eps)

            std = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=std, op=nl.sqrt, data=var)
            nisa.reciprocal(dst=rstd, data=std)

            # ``sqrt`` and ``reciprocal`` are hardware approximations (~1e-5
            # relative error), which is looser than the fp32 tolerance this
            # operator is verified against. Two Newton-Raphson steps of
            # ``r <- r * (1.5 - 0.5 * var * r^2)`` restore full fp32 accuracy;
            # they run on [P, 1] vectors, so the cost is negligible.
            half_var = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=half_var, data=var, op0=nl.multiply, operand0=0.5)

            correction = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            for _newton_step in range(2):
                nisa.tensor_tensor(dst=correction, data1=rstd, data2=rstd, op=nl.multiply)
                nisa.tensor_tensor(dst=correction, data1=correction, data2=half_var,
                                   op=nl.multiply)
                # correction = 1.5 - 0.5 * var * rstd^2
                nisa.tensor_scalar(dst=correction, data=correction,
                                   op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.5)
                nisa.tensor_tensor(dst=rstd, data1=rstd, data2=correction, op=nl.multiply)

            # (x - mean) * rstd == x * rstd + neg_mean_rstd, which keeps the
            # fused tensor_scalar in the (multiply, add) form supported by both
            # the Vector and the Scalar engine.
            neg_mean_rstd = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=neg_mean_rstd, data1=mean, data2=rstd, op=nl.multiply)
            nisa.tensor_scalar(dst=neg_mean_rstd, data=neg_mean_rstd,
                               op0=nl.multiply, operand0=-1.0)

            # ---- pass 2: y = normalized * weight + bias ---------------------
            for col_tile in nl.affine_range(n_col_tiles):
                col_start = col_tile * FREE_TILE
                col_size = min(FREE_TILE, n_cols - col_start)
                col_end = col_start + col_size

                x_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=x_tile, src=a_input[row_start:row_end, col_start:col_end])

                normalized = nl.ndarray((row_size, col_size), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=normalized, data=x_tile,
                                   op0=nl.multiply, operand0=rstd,
                                   op1=nl.add, operand1=neg_mean_rstd)
                nisa.tensor_tensor(dst=normalized, data1=normalized,
                                   data2=weight_bcast[0:row_size, col_start:col_end],
                                   op=nl.multiply)

                y_tile = nl.ndarray((row_size, col_size), dtype=a_input.dtype, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=y_tile, data1=normalized,
                                   data2=bias_bcast[0:row_size, col_start:col_end],
                                   op=nl.add)

                nisa.dma_copy(dst=out_hbm[row_start:row_end, col_start:col_end], src=y_tile)

        return out_hbm


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5,
        block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:

    if x.dtype == torch.int8:
        raise NotImplementedError("layernorm NKI: int8 not supported")

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    if not x_2d.is_contiguous():
        x_2d = x_2d.contiguous()

    weight_2d = weight.reshape(1, -1)
    bias_2d = bias.reshape(1, -1)
    out_2d = layernorm_kernel(x_2d, weight_2d, bias_2d, float(eps))
    return out_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    return None
