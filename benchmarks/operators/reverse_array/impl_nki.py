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
    def reverse_kernel(a_input):
        n = a_input.shape[0]

        num_blocks = (n + (PMAX - 1)) // PMAX

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(a_input.shape[1])[None, :]
            
            mask = partition_index < (n - offset)

            a_tile = nl.load(a_input[offset + partition_index, free_dim_index], mask=mask)
            
            nl.store(hbm_result_tile[n - 1 - (offset + partition_index), free_dim_index], value=a_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    x_2d = x.reshape(-1, 1)
    result = reverse_kernel(x_2d)
    return result.reshape(-1)