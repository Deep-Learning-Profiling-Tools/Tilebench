import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
    TILE_C = 512
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def batch_norm_kernel(a_input, gamma_input, beta_input, eps):
        # a_input: (N, C). gamma/beta: (1, C).
        # BatchNorm reduces over the batch dim N, which sits on the *partition*
        # axis here (unlike LayerNorm's free-axis reduction) -- nl reductions
        # only work along the free axis, so per-channel sums are instead
        # produced with the Tensor Engine: ones[K,1].T @ x[K,TILE_C] sums the K
        # (<=128) rows on the partition axis into a (1, TILE_C) row, accumulated
        # in PSUM across all N tiles. Pass 2 then broadcasts the resulting
        # (1, TILE_C) mean/rstd/gamma/beta rows across partitions (as in
        # LayerNorm) and normalizes.
        N, C = a_input.shape

        num_n_blocks = (N + (PMAX - 1)) // PMAX
        num_c_blocks = (C + TILE_C - 1) // TILE_C

        hbm_result = nl.ndarray((N, C), dtype=a_input.dtype, buffer=nl.hbm)

        ones_tile = nl.full((PMAX, 1), 1.0, dtype=nl.float32, buffer=nl.sbuf)

        for ci in range(num_c_blocks):
            c_offset = ci * TILE_C
            free_dim_index = nl.arange(TILE_C)[None, :]
            mask_c = free_dim_index < (C - c_offset)

            sum_x_psum = nl.zeros((1, TILE_C), dtype=nl.float32, buffer=nl.psum)
            sum_x2_psum = nl.zeros((1, TILE_C), dtype=nl.float32, buffer=nl.psum)

            for ni in nl.sequential_range(num_n_blocks):
                n_offset = ni * PMAX
                partition_index = nl.arange(PMAX)[:, None]
                mask_p = partition_index < (N - n_offset)
                mask = mask_p & mask_c

                x_tile = nl.load(a_input[n_offset + partition_index, c_offset + free_dim_index],
                                  mask=mask, dtype=nl.float32)
                zero_tile = nl.zeros(x_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
                x_safe = nl.where(mask, x_tile, zero_tile)

                sum_x_psum += nisa.nc_matmul(ones_tile, x_safe)
                sum_x2_psum += nisa.nc_matmul(ones_tile, nl.multiply(x_safe, x_safe))

            sum_x = nl.copy(sum_x_psum, dtype=nl.float32)
            sum_x2 = nl.copy(sum_x2_psum, dtype=nl.float32)

            mean = nl.divide(sum_x, float(N))
            mean_sq = nl.divide(sum_x2, float(N))
            var = nl.subtract(mean_sq, nl.multiply(mean, mean))
            rstd = nl.rsqrt(nl.add(var, eps, dtype=nl.float32))

            w_partition_index = nl.arange(1)[:, None]
            gamma_tile = nl.load(gamma_input[w_partition_index, c_offset + free_dim_index],
                                  mask=mask_c, dtype=nl.float32)
            beta_tile = nl.load(beta_input[w_partition_index, c_offset + free_dim_index],
                                 mask=mask_c, dtype=nl.float32)

            mean_b = nl.broadcast_to(mean, shape=(PMAX, TILE_C))
            rstd_b = nl.broadcast_to(rstd, shape=(PMAX, TILE_C))
            gamma_b = nl.broadcast_to(gamma_tile, shape=(PMAX, TILE_C))
            beta_b = nl.broadcast_to(beta_tile, shape=(PMAX, TILE_C))

            for ni in range(num_n_blocks):
                n_offset = ni * PMAX
                partition_index = nl.arange(PMAX)[:, None]
                mask_p = partition_index < (N - n_offset)
                mask = mask_p & mask_c

                x_tile = nl.load(a_input[n_offset + partition_index, c_offset + free_dim_index],
                                  mask=mask, dtype=nl.float32)

                centered = nl.subtract(x_tile, mean_b)
                normalized = nl.multiply(centered, rstd_b)
                scaled = nl.multiply(normalized, gamma_b)
                result_fp32 = nl.add(scaled, beta_b)
                result_tile = nl.add(result_fp32, 0.0, dtype=a_input.dtype)

                nl.store(hbm_result[n_offset + partition_index, c_offset + free_dim_index],
                         value=result_tile, mask=mask)

        return hbm_result


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if input.dtype == torch.int8:
        raise NotImplementedError("batch_normalization NKI: int8 not supported")
    gamma_2d = gamma.reshape(1, -1)
    beta_2d = beta.reshape(1, -1)
    return batch_norm_kernel(input, gamma_2d, beta_2d, eps)


def get_last_config() -> dict | None:
    return None
