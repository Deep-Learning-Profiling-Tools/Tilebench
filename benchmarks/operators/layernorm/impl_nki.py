import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def layernorm_kernel(a_input, weight_input, bias_input, eps):
        free_tile_size = 2048

        num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX
        num_free_blocks = (a_input.shape[1] + free_tile_size - 1) // free_tile_size
        n_cols = a_input.shape[1]

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (a_input.shape[0] - offset)

            # Pass 1: accumulate sum(x) and sum(x^2) per row (fp32 accumulators).
            sum_x = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            sum_x2 = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
                mask_f = free_dim_index < (n_cols - free_offset)
                mask = mask_p & mask_f

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)

                zero_tile = nl.zeros(a_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
                a_safe = nl.where(mask, a_tile, zero_tile)

                sum_x[...] = nl.add(sum_x, nl.sum(a_safe, axis=1, keepdims=True))
                sq_tile = nl.multiply(a_safe, a_safe)
                sum_x2[...] = nl.add(sum_x2, nl.sum(sq_tile, axis=1, keepdims=True))

            mean = nl.divide(sum_x, float(n_cols))
            var = nl.subtract(nl.divide(sum_x2, float(n_cols)), nl.multiply(mean, mean))

            val = nl.add(var, eps, dtype=nl.float32)
            half_val = nl.multiply(val, 0.5)
            rstd0 = nl.rsqrt(val)
            corr1 = nl.subtract(1.5, nl.multiply(half_val, nl.multiply(rstd0, rstd0)))
            rstd1 = nl.multiply(rstd0, corr1)
            corr2 = nl.subtract(1.5, nl.multiply(half_val, nl.multiply(rstd1, rstd1)))
            rstd = nl.multiply(rstd1, corr2)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
                mask_f = free_dim_index < (n_cols - free_offset)
                mask = mask_p & mask_f

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)

                w_partition_index = nl.arange(1)[:, None]
                w_tile = nl.load(weight_input[w_partition_index, free_offset + free_dim_index], mask=mask_f, dtype=nl.float32)
                b_tile = nl.load(bias_input[w_partition_index, free_offset + free_dim_index], mask=mask_f, dtype=nl.float32)
                w_broadcast = nl.broadcast_to(w_tile, shape=(PMAX, free_tile_size))
                b_broadcast = nl.broadcast_to(b_tile, shape=(PMAX, free_tile_size))

                centered = nl.subtract(a_tile, mean)
                normalized = nl.multiply(centered, rstd)
                scaled = nl.multiply(normalized, w_broadcast)
                result_tile = nl.add(scaled, b_broadcast, dtype=a_input.dtype)

                nl.store(hbm_result_tile[offset + partition_index, free_offset + free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile


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
    out_2d = layernorm_kernel(x_2d, weight_2d, bias_2d, eps)
    return out_2d.reshape(orig_shape)


def get_last_config() -> dict | None:
    return None
