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
    def mean_rowwise_kernel(a_input):
        m, n = a_input.shape

        num_blocks = (m + (PMAX - 1)) // PMAX

        hbm_result_tile = nl.ndarray((m, 1), dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i*PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(a_input.shape[1])[None, :]
            
            mask = partition_index < (a_input.shape[0] - offset)

            a_tile = nl.load(a_input[offset + partition_index, free_dim_index], mask=mask, dtype=nl.float32)
            
            row_sum = nl.sum(a_tile, axis=1)
            row_mean = nl.divide(row_sum, n)
            
            nl.store(hbm_result_tile[offset + partition_index, nl.arange(1)[None, :]], value=row_mean, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, dim: int = 1, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("mean reduction NKI: int8 not supported")
    
    result = mean_rowwise_kernel(x)
    return result.reshape(-1) 

def get_last_config() -> dict | None:
    return None