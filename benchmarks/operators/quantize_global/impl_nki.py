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
    def quantize_global_kernel(a_input):
        free_tile_size = 16384

        num_blocks = (a_input.shape[0] + (PMAX - 1)) // PMAX
        num_free_blocks = (a_input.shape[1] + free_tile_size - 1) // free_tile_size

        hbm_result_tile = nl.ndarray(a_input.shape, dtype=nl.float16, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            mask_p = partition_index < (a_input.shape[0] - offset)

            for j in range(num_free_blocks):
                free_offset = j * free_tile_size
                free_dim_index = nl.arange(free_tile_size)[None, :]
                mask_f = free_dim_index < (a_input.shape[1] - free_offset)
                mask = mask_p & mask_f

                a_tile = nl.load(a_input[offset + partition_index, free_offset + free_dim_index], mask=mask)
                result_tile = nl.add(a_tile, 0.0, dtype=nl.float16)

                nl.store(hbm_result_tile[offset + partition_index, free_offset + free_dim_index], value=result_tile, mask=mask)

        return hbm_result_tile


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    result = quantize_global_kernel(x_2d)
    return result.reshape(-1)[:n]


def get_last_config() -> dict | None:
    return None
