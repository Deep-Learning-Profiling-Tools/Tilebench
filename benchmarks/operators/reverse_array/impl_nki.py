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
    def reverse_kernel(a_input):
        num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX
        free_tile_size = 16384
        num_free_blocks = (a_input.shape[1] + free_tile_size - 1) // free_tile_size

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        total_rows = a_input.shape[0]
        total_cols = a_input.shape[1] 
        
        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (total_rows - offset)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
                mask_f = free_dim_index < (total_cols - free_offset)
                mask = mask_p & mask_f

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask)

                # reverse destination: element at (p, f) goes to (total_rows-1-p, total_cols-1-f)
                rev_p = total_rows - 1 - (offset + partition_index)
                rev_f = total_cols - 1 - (free_offset + free_dim_index)

                nl.store(hbm_result_tile[rev_p, rev_f], value=a_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    result = reverse_kernel(x_2d)

    pad_count = padded_size - n

    return result.reshape(-1)[pad_count:]

def get_last_config() -> dict | None:
    return None