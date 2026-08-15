import torch

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

@nki.jit
def dropout_kernel(x_input, x_keep_input, p):
    """Dropout scaling: out = (x / (1 - p)) * x_keep, tiled over partition and free dims."""
    P, F = x_input.shape

    num_blocks = (P + PMAX - 1) // PMAX

    free_tile_size = 16384
    num_free_blocks = (F + free_tile_size - 1) // free_tile_size

    # nl.divide is not a supported tensor_scalar operator, so scale by the reciprocal.
    keep_scale = 1.0 / (1.0 - p)

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

            x_keep_tile = nl.ndarray((p_sz, f_sz), dtype=x_keep_input.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_keep_tile, src=x_keep_input[p_start:p_end, f_start:f_end])

            # scaled = x / (1 - p)
            scaled_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=scaled_tile, data=x_tile, op0=nl.multiply,
                               operand0=keep_scale)

            # out = scaled * x_keep
            result_tile = nl.ndarray((p_sz, f_sz), dtype=x_input.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=result_tile, data1=scaled_tile, data2=x_keep_tile,
                               op=nl.multiply)

            nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, f_start:f_end], src=result_tile)

    return hbm_result_tile

def run(x: torch.Tensor, x_keep: torch.Tensor, p: float, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    if x.dtype == torch.int8:
        raise NotImplementedError("dropout NKI: int8 not supported")

    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))
        x_keep = torch.nn.functional.pad(x_keep, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    x_keep_2d = x_keep.reshape(PMAX, free_dim)
    result = dropout_kernel(x_2d, x_keep_2d, p)

    return result.reshape(-1)[:n]

def get_last_config() -> dict | None:
    return None
