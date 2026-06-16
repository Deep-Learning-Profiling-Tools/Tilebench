import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
    FMAX_SBUF = 64000
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def relu_kernel(a_input):
        free_dim = min(a_input.shape[1], FMAX_SBUF)

        num_free_blocks = (a_input.shape[1] + (free_dim - 1)) // free_dim

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        partition_index = nl.arange(PMAX)[:, None]

        for j in range(num_blocks):
            free_offset = j * free_dim

            free_dim_index = nl.arange(free_dim)[None, :]
            
            free_mask = free_dim_index < (a_input.shape[1] - free_offset)

            a_tile = nl.load(a_input[partition_index, free_offset + free_dim_index], mask=free_mask)

            result_tile = nl.maximum(a_tile, 0, mask=free_mask)
            
            nl.store(hbm_result_tile[partition_index, free_offset + free_dim_index], value=result_tile, mask=free_mask)

        return hbm_result_tile

def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    result = relu_kernel(x_2d)

    return result.reshape(-1)[:n]

def get_last_config() -> dict | None:
    return None