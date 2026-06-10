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
    def matrix_copy_kernel(a_input):
        num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(a_input.shape[1])[None, :]
            
            mask = partition_index < (a_input.shape[0] - offset)

            a_tile = nl.load(a_input[offset + partition_index, free_dim_index], mask=mask)
            
            nl.store(hbm_result_tile[offset + partition_index, free_dim_index], value=a_tile, mask=mask)

        return hbm_result_tile

def run(A: torch.Tensor, N: int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    return matrix_copy_kernel(A)

def get_last_config() -> dict | None:
    return None

