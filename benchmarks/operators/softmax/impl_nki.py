import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
    FREE_CAP = 8192
    FALLBACK_TILE = 2048

except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def softmax_online_kernel(a_input):
        n_rows = a_input.shape[0]
        n_cols = a_input.shape[1]
        NEG_INF = -3.0e38

        num_blocks = (n_rows + (PMAX - 1)) // PMAX
        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        _cap = 2048 if a_input.dtype == nl.float32 else FREE_CAP
        if n_cols <= _cap:
            for i in range(num_blocks):
                offset = i * PMAX
                partition_index = nl.arange(PMAX)[:, None]
                free_index = nl.arange(n_cols)[None, :]
                mask = (partition_index < (n_rows - offset)) & (free_index < n_cols)

                a_tile = nl.load(a_input[offset + partition_index, free_index], mask=mask, dtype=nl.float32)
                
                row_max = nl.max(a_tile, axis=1, keepdims=True)
                exp_tile = nl.exp(nl.subtract(a_tile, row_max))
                row_sum = nl.sum(exp_tile, axis=1, keepdims=True)
                
                result_fp32 = nl.divide(exp_tile, row_sum)
                result_tile = nl.add(result_fp32, 0.0, dtype=a_input.dtype)
                nl.store(hbm_result_tile[offset + partition_index, free_index], value=result_tile, mask=mask)
            return hbm_result_tile

        num_free_blocks = (n_cols + FALLBACK_TILE - 1) // FALLBACK_TILE
        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (n_rows - offset)

            row_max = nl.full((PMAX, 1), NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
            for j in range(num_free_blocks):
                free_offset = j * FALLBACK_TILE
                free_dim_index = nl.arange(FALLBACK_TILE)[None, :]
                mask = mask_p & (free_dim_index < (n_cols - free_offset))
                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)
                neg_fill = nl.full(a_tile.shape, NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
                a_for_max = nl.where(mask, a_tile, neg_fill)
                row_max[...] = nl.maximum(row_max, nl.max(a_for_max, axis=1, keepdims=True))

            row_sum = nl.zeros((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            for j in range(num_free_blocks):
                free_offset = j * FALLBACK_TILE
                free_dim_index = nl.arange(FALLBACK_TILE)[None, :]
                mask = mask_p & (free_dim_index < (n_cols - free_offset))
                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)
                exp_tile = nl.exp(nl.subtract(a_tile, row_max))
                zero_tile = nl.zeros(exp_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
                exp_safe = nl.where(mask, exp_tile, zero_tile)
                row_sum[...] = nl.add(row_sum, nl.sum(exp_safe, axis=1, keepdims=True))

            for j in range(num_free_blocks):
                free_offset = j * FALLBACK_TILE
                free_dim_index = nl.arange(FALLBACK_TILE)[None, :]
                mask = mask_p & (free_dim_index < (n_cols - free_offset))
                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask, dtype=nl.float32)
                exp_tile = nl.exp(nl.subtract(a_tile, row_max))
                result_fp32 = nl.divide(exp_tile, row_sum)
                result_tile = nl.add(result_fp32, 0.0, dtype=a_input.dtype)
                nl.store(hbm_result_tile[offset + partition_index, free_offset + free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    return softmax_online_kernel(x)

def get_last_config() -> dict | None:
    return None
