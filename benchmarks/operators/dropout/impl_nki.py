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
    def dropout_kernel(x_input, x_keep_input, p):
        num_blocks = (x_input.shape[0] + (PMAX - 1)) // PMAX

        hbm_result_tile = nl.ndarray(x_input.shape, dtype=x_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(x_input.shape[1])[None, :]
            
            mask = partition_index < (x_input.shape[0] - offset)

            x_tile = nl.load(x_input[offset + partition_index, free_dim_index], mask=mask)
            x_keep_tile = nl.load(x_keep_input[offset + partition_index, free_dim_index], mask=mask)

            scaled = nl.divide(x_tile, (1.0 - p), mask=mask)

            result_tile = nl.multiply(scaled, x_keep_tile, mask=mask)
            
            nl.store(hbm_result_tile[offset + partition_index, free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, x_keep: torch.Tensor, p: float, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("dropout NKI: int8 not supported")
    x_2d = x.reshape(-1, 1)
    x_keep_2d = x_keep.reshape(-1, 1)
    result = dropout_kernel(x_2d, x_keep_2d, p)
    return result.reshape(-1)