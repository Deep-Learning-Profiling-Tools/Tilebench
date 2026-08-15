import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

@nki.jit
def matrix_copy_kernel(a_input):
    """Element-wise matrix copy: out = in, tiled over rows (partition) and columns (free)."""
    P, F = a_input.shape

    num_blocks = (P + PMAX - 1) // PMAX

    free_tile_size = 16384
    num_free_blocks = (F + free_tile_size - 1) // free_tile_size

    hbm_result_tile = nl.ndarray(a_input.shape, dtype=a_input.dtype, buffer=nl.shared_hbm)

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

            nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, f_start:f_end], src=a_tile)

    return hbm_result_tile

def run(A: torch.Tensor, N: int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    return matrix_copy_kernel(A)

def get_last_config() -> dict | None:
    return None
