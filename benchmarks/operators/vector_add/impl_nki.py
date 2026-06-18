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
    def add_kernel(a_input, b_input):
        assert a_input.shape == b_input.shape
        assert a_input.dtype == b_input.dtype

        num_blocks = (a_input.shape[0] + PMAX - 1) // PMAX

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(a_input.shape[1])[None, :]

            mask = partition_index < (a_input.shape[0] - offset)

            a_tile = nl.load(a_input[offset + partition_index, free_dim_index], mask=mask)
            b_tile = nl.load(b_input[offset + partition_index, free_dim_index], mask=mask)

            result_tile = nl.add(a_tile, b_tile, mask=mask)

            nl.store(hbm_result_tile[offset + partition_index, free_dim_index], value=result_tile, mask=mask)
        
        return hbm_result_tile

def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))
        y = torch.nn.functional.pad(y, (0, padded_size - n))
    
    x_2d = x.reshape(PMAX, free_dim)
    y_2d = y.reshape(PMAX, free_dim)
    result = add_kernel(x_2d, y_2d)
    return result.reshape(-1)[:n]


def get_last_config() -> dict | None:
    return None

    
