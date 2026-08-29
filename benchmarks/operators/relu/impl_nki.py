from types import SimpleNamespace

import torch

from core.nki_autotune import NkiAutotuner

try:
    import nki
    import nki.language as nl
    import nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

@nki.jit
def relu_kernel(a_input, block_size):
    P, F = a_input.shape

    num_blocks = (P + PMAX - 1) // PMAX

    free_tile_size = block_size
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

            result_tile = nl.ndarray((p_sz, f_sz), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=result_tile, data=a_tile, op0=nl.maximum, operand0=0)

            nisa.dma_copy(dst=hbm_result_tile[p_start:p_end, f_start:f_end], src=result_tile)

    return hbm_result_tile

_DEFAULT_CONFIG = SimpleNamespace(block_size=16384)
_SEARCH_SPACE = [SimpleNamespace(block_size=b) for b in (2048, 4096, 8192, 16384)]
_tuner = NkiAutotuner(relu_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, block_size: int = 1024, autotune=False, **kwargs) -> torch.Tensor:
    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x_2d.shape), str(x_2d.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_2d, cfg.block_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = relu_kernel(x_2d, cfg.block_size)

    return result.reshape(-1)[:n]

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None
