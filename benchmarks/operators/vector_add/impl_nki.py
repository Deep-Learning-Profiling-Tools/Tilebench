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

NUM_CORES = 2  # matches the repo-wide NEURON_LOGICAL_NC_CONFIG=2 ("LNC2") convention


@nki.jit
def add_kernel(a_input, b_input, free_tile_size):
    """a_input/b_input: [PMAX, F] (see run() -- always exactly PMAX rows).

    Sharded across NUM_CORES physical NeuronCores via NKI's LNC mechanism:
    run() launches this as add_kernel[NUM_CORES](...), and each of the
    NUM_CORES program instances handles an independent 1/NUM_CORES slice of
    the free dimension, selected via nl.program_id(0) (a runtime value, so
    the per-core offset uses nl.ds()/.ap(offset=...) dynamic addressing
    rather than Python-level slicing -- see AWS's LNC guide's
    "divide work between multiple cores" pattern). Without this, LNC2 alone
    only makes the second physical core addressable; it does not shard a
    kernel's work onto it (confirmed via profiling: the un-sharded kernel
    used only 16 of 32 available DMA engines).

    Returns a flat [PMAX * F] HBM tensor rather than [PMAX, F]: writing each
    tile's result directly at its final flat offset (via .ap()) lets run()
    hand the result straight back to the caller with no reshape/slice, which
    otherwise compiles to two extra full-size HBM materializations after the
    custom call (measured: ~52% of total wall time for large N) instead of a
    free view/no-op.
    """
    assert a_input.shape == b_input.shape
    assert a_input.dtype == b_input.dtype

    P, F = a_input.shape
    assert F % NUM_CORES == 0, f"free dim {F} must be divisible by NUM_CORES={NUM_CORES}"
    core_F = F // NUM_CORES
    num_free_blocks = (core_F + free_tile_size - 1) // free_tile_size

    hbm_result_tile = nl.ndarray((P * F,), dtype=a_input.dtype, buffer=nl.shared_hbm)

    core_offset = nl.program_id(0) * core_F

    for j in range(num_free_blocks):
        f_start = j * free_tile_size
        f_sz = min(free_tile_size, core_F - f_start)

        a_tile = nl.ndarray((P, f_sz), dtype=a_input.dtype, buffer=nl.sbuf)
        b_tile = nl.ndarray((P, f_sz), dtype=b_input.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_tile, src=a_input[:, nl.ds(core_offset + f_start, f_sz)])
        nisa.dma_copy(dst=b_tile, src=b_input[:, nl.ds(core_offset + f_start, f_sz)])

        result_tile = nl.add(a_tile, b_tile)

        # Row p of this tile belongs at flat offset p*F + core_offset + f_start.
        nisa.dma_copy(
            dst=hbm_result_tile.ap(pattern=[[F, P], [1, f_sz]], offset=core_offset + f_start),
            src=result_tile,
        )

    return hbm_result_tile


_DEFAULT_CONFIG = SimpleNamespace(free_tile_size=16384)
_SEARCH_SPACE = [SimpleNamespace(free_tile_size=t)
                 for t in (1024, 2048, 4096, 8192, 16384)]
_sharded_kernel = add_kernel[NUM_CORES] if nki is not None else None
_tuner = NkiAutotuner(_sharded_kernel) if nki is not None else None
_last_autotune_config: dict = {}


def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    n = x.numel()
    free_dim = (n + (PMAX - 1)) // PMAX
    # add_kernel splits the free dim evenly across NUM_CORES cores.
    free_dim = ((free_dim + NUM_CORES - 1) // NUM_CORES) * NUM_CORES
    padded_size = PMAX * free_dim

    if padded_size > n:
        x = torch.nn.functional.pad(x, (0, padded_size - n))
        y = torch.nn.functional.pad(y, (0, padded_size - n))

    x_2d = x.reshape(PMAX, free_dim)
    y_2d = y.reshape(PMAX, free_dim)
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(tuple(x_2d.shape), str(x_2d.dtype)),
            search_space=_SEARCH_SPACE,
            args_fn=lambda cfg: (x_2d, y_2d, cfg.free_tile_size),
        )
        _last_autotune_config.clear()
        _last_autotune_config.update(vars(cfg))
    else:
        cfg = _DEFAULT_CONFIG
    result = _sharded_kernel(x_2d, y_2d, cfg.free_tile_size)
    return result[:n] if padded_size > n else result


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) or None

    
