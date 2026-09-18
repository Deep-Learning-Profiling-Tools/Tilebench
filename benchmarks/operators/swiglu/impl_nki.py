import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

@nki.jit
def swiglu_kernel(x_input, y_input):
    """SwiGLU: out = silu(x) * y, tiled over partition and free dims."""
    P, F = x_input.shape

    num_blocks = (P + PMAX - 1) // PMAX

    free_tile_size = 16384
    num_free_blocks = (F + free_tile_size - 1) // free_tile_size

    hbm_result_tile = nl.ndarray(x_input.shape, dtype=x_input.dtype, buffer=nl.shared_hbm)

    for i in range(num_blocks):
        p_start = i * PMAX
        p_end = min(p_start + PMAX, P)
        p_sz = p_end - p_start

        for j in range(num_free_blocks):
            f_start = j * free_tile_size
            f_end = min(f_start + free_tile_size, F)
            f_sz = f_end - f_start

            x_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_tile, src=x_input[p_start:p_end, f_start:f_end])

            y_tile = nl.ndarray((p_sz, f_sz), dtype=y_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=y_tile, src=y_input[p_start:p_end, f_start:f_end])

            # silu_x = silu(x)
            silu_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.activation(dst=silu_tile, data=x_tile, op=nl.silu)

            # out = silu_x * y
            result_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=result_tile, data1=silu_tile, data2=y_tile,
                               op=nl.multiply)

            nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, f_start:f_end], src=result_tile)

    return hbm_result_tile

def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    return swiglu_kernel(x, y)

def get_last_config() -> dict | None:
    return None
