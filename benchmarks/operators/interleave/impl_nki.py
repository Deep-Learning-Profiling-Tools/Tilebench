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
    def interleave_kernel(a_input, b_input):
        num_rows = a_input.shape[0]
        num_cols = a_input.shape[1]
        num_blocks = (num_rows + (PMAX - 1)) // PMAX
        free_tile_size = 16384
        num_free_blocks = (num_cols + free_tile_size - 1) // free_tile_size

        hbm_shape = (num_rows, num_cols * 2)
        hbm_result_tile = nl.ndarray(hbm_shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX
            partition_index = nl.arange(PMAX)[:, None]  
            mask_p = partition_index < (num_rows - offset)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
                mask_f = free_dim_index < (num_cols - free_offset)
                mask = mask_p & mask_f

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask)
                b_tile = nl.load(b_input[offset + partition_index, free_offset + free_dim_index], mask=mask)

                col_index_a = (free_offset + free_dim_index) * 2
                col_index_b = (free_offset + free_dim_index) * 2 + 1
                
                nl.store(hbm_result_tile[offset + partition_index, col_index_a], value=a_tile, mask=mask)
                nl.store(hbm_result_tile[offset + partition_index, col_index_b], value=b_tile, mask=mask)
            
        return hbm_result_tile

def run(a: torch.Tensor, b: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        a = torch.nn.functional.pad(a, (0, padded_size - n))
        b = torch.nn.functional.pad(b, (0, padded_size - n))

    a_2d = a.reshape(PMAX, free_dim)
    b_2d = b.reshape(PMAX, free_dim)
    result = interleave_kernel(a_2d, b_2d)
    return result.reshape(-1)[:2 * n]

def get_last_config() -> dict | None:
    return None