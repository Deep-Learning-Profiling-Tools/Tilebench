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
    def swiglu_kernel(x_input, y_input):
        num_blocks = (x_input.shape[0] + PMAX - 1) // PMAX

        hbm_result_tile = nl.ndarray(x_input.shape, dtype=x_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(x_input.shape[1])[None, :]

            mask = partition_index < (x_input.shape[0] - offset)

            x_tile = nl.load(x_input[offset + partition_index, free_dim_index], mask=mask)
            y_tile = nl.load(y_input[offset + partition_index, free_dim_index], mask=mask)

            silu_x = nl.silu(x_tile, mask=mask)
            result_tile = nl.multiply(silu_x, y_tile, mask=mask)

            nl.store(hbm_result_tile[offset + partition_index, free_dim_index], value=result_tile, mask=mask)
        
        return hbm_result_tile

def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    return swiglu_kernel(x, y)

def get_last_config() -> dict | None:
    return None
