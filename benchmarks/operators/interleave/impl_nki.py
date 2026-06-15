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
    def interleave_kernel(a_input, b_input):
        num_rows = a_input.shape[0]
        num_cols = a_input.shape[1]

        num_blocks = (num_rows + (PMAX - 1)) // PMAX

        hbm_shape = (num_rows, num_cols * 2)
        hbm_result_tile = nl.ndarray(hbm_shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(num_cols)[None, :]
            
            mask = partition_index < (num_rows - offset)

            a_tile = nl.load(a_input[offset + partition_index, free_dim_index], mask=mask)
            b_tile = nl.load(b_input[offset + partition_index, free_dim_index], mask=mask)

            col_index_a = free_dim_index * 2
            nl.store(hbm_result_tile[offset + partition_index, col_index_a], value=a_tile, mask=mask)

            col_index_b = (free_dim_index * 2) + 1
            nl.store(hbm_result_tile[offset + partition_index, col_index_b], value=b_tile, mask=mask)
            
        return hbm_result_tile

def run(a: torch.Tensor, b: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    a_2d = a.reshape(-1, 1)
    b_2d = b.reshape(-1, 1)
    result = interleave_kernel(a_2d, b_2d)
    return result.reshape(-1)