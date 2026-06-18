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
    def reverse_kernel(a_input):
        n = a_input.shape[0]

        num_blocks = (n + (PMAX - 1)) // PMAX

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX

            partition_index = nl.arange(PMAX)[:, None]
            free_dim_index = nl.arange(a_input.shape[1])[None, :]
            
            mask = partition_index < (n - offset)

            a_tile = nl.load(a_input[offset + partition_index, free_dim_index], mask=mask)

            rev_partition = PMAX - 1 - partition_index
            rev_free = a_input.shape[1] - 1 - free_dim_index
            
            nl.store(hbm_result_tile[rev_partition, rev_free], value=a_tile, mask=mask)

        return hbm_result_tile

def run(x: torch.Tensor, n: int, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    result = reverse_kernel(x_2d)

    pad_count = padded_size - n

    return result.reshape(-1)[pad_count:]

def get_last_config() -> dict | None:
    return None