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
    def dropout_kernel(x_input, x_keep_input, p):
        free_tile_size = 16384
        
        num_blocks = (x_input.shape[0] + (PMAX - 1)) // PMAX
        num_free_blocks = (x_input.shape[1] + free_tile_size - 1) // free_tile_size
        
        hbm_result_tile = nl.ndarray(x_input.shape, dtype=x_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (x_input.shape[0] - offset)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
            
                mask_f = free_dim_index < (x_input.shape[1] - free_offset)
                mask = mask_p & mask_f

                x_tile = nl.load(x_input[offset + partition_index, free_offset + free_dim_index], mask=mask)
                x_keep_tile = nl.load(x_keep_input[offset + partition_index, free_offset + free_dim_index], mask=mask)

                scaled = nl.divide(x_tile, (1.0 - p), mask=mask)
                result_tile = nl.multiply(scaled, x_keep_tile, mask=mask)
            
                nl.store(hbm_result_tile[offset + partition_index, free_offset + free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, x_keep: torch.Tensor, p: float, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("dropout NKI: int8 not supported")
    
    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))
        x_keep = torch.nn.functional.pad(x_keep, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    x_keep_2d = x_keep.reshape(PMAX, free_dim)
    result = dropout_kernel(x_2d, x_keep_2d, p)
    
    return result.reshape(-1)[:n]

def get_last_config() -> dict | None:
    return None
