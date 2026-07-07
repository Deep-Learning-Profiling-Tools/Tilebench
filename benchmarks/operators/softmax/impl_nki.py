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
    def softmax_online_kernel(a_input):
        free_tile_size = 16384

        num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX
        num_free_blocks = (a_input.shape[1] + free_tile_size - 1) // free_tile_size

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (a_input.shape[0] - offset)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]

                mask_f = free_dim_index < (a_input.shape[1] - free_offset)
                mask = mask_p & mask_f

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask)
            
                row_max = nl.max(a_tile, axis=1)
                shifted = nl.subtract(a_tile, row_max)
                exp_tile = nl.exp(shifted)
                row_sum = nl.sum(exp_tile, axis=1)
                result_tile = nl.divide(exp_tile, row_sum)
            
            nl.store(hbm_result_tile[offset + partition_index, free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    return softmax_online_kernel(x)

def get_last_config() -> dict | None:
    return None