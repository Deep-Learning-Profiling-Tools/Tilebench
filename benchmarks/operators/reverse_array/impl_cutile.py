from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [1024, 2048, 4096, 8192]
    for occ in [4, 8, 16]
]


@ct.kernel
def reverse_kernel(x_ptr, out_ptr, N, TILE: ConstInt):
    bid = ct.bid(0)
    offsets = ct.arange(TILE, dtype=ct.int32) + bid * TILE
    rev_indices = N - 1 - offsets
    vals = ct.gather(x_ptr, rev_indices, padding_value=0)
    ct.store(out_ptr, index=(bid,), tile=vals)


_tuner = CutileAutotuner(reverse_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((N + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (input, output, N, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile": cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = ((N + cfg.tile - 1) // cfg.tile, 1, 1)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel, (input, output, N, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
