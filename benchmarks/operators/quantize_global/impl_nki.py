import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

@nki.jit
def quantize_global_kernel(a_input):
    """Global quantization to fp16: out = a.to(float16), tiled over partition and free dims."""
    P, F = a_input.shape

    num_blocks = (P + PMAX - 1) // PMAX

    free_tile_size = 16384
    num_free_blocks = (F + free_tile_size - 1) // free_tile_size

    hbm_result_tile = nl.ndarray(a_input.shape, dtype=nl.float16, buffer=nl.shared_hbm)

    for i in range(num_blocks):
        p_start = i * PMAX
        p_end = min(p_start + PMAX, P)
        p_sz = p_end - p_start

        for j in range(num_free_blocks):
            f_start = j * free_tile_size
            f_end = min(f_start + free_tile_size, F)
            f_sz = f_end - f_start

            a_tile = nl.ndarray((p_sz, f_sz), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a_input[p_start:p_end, f_start:f_end])

            # tensor_copy performs the dtype cast to the fp16 destination tile.
            result_tile = nl.ndarray((p_sz, f_sz), dtype=nl.float16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_tile, src=a_tile)

            nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, f_start:f_end], src=result_tile)

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
