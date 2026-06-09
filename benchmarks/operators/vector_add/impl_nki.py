import torch
from torch_xla.core import xla_model as xm

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
            free_dim_index = nl.arange(1)[None, :]

            mask = partition_index < (a_input.shape[0] - offset)

            a_tile = nl.load(a_input[offset + partition_index, free_dim_index], mask=mask)
            b_tile = nl.load(b_input[offset + partition_index, free_dim_index], mask=mask)

            # nl.add handles partial tiles correctly
            result_tile = nl.add(a_tile, b_tile, mask=mask)

            nl.store(hbm_result_tile[offset + partition_index, free_dim_index], value=result_tile, mask=mask)
        
        return hbm_result_tile

def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    x_2d = x.reshape(-1, 1)
    y_2d = y.reshape(-1, 1)
    result = add_kernel(x_2d, y_2d)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None

    
