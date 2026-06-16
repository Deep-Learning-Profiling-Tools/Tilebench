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
    def leaky_relu_kernel(a_input):
        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        partition_index = nl.arange(PMAX)[:, None]
        free_dim_index = nl.arange(a_input.shape[1])[None, :]

        a_tile = nl.load(a_input[partition_index, free_dim_index])

        scaled_tile = nl.multiply(a_tile, 0.01)

        result_tile = nl.maximum(a_tile, scaled_tile)
            
        nl.store(hbm_result_tile[partition_index, free_dim_index], value=result_tile)

        return hbm_result_tile

def run(x: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("leaky_relu NKI: int8 not supported")
    
    free_dim = (n + (PMAX -1)) // PMAX
    padded_size = PMAX * free_dim
    
    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    result = leaky_relu_kernel(x_2d)

    return result.reshape(-1)[:n]

def get_last_config() -> dict | None:
    return None