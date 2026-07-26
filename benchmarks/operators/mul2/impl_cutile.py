from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)


@ct.kernel
def mul2_kernel(x_ptr, output_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,))
    y_tile = x_tile * 2
    ct.store(output_ptr, index=(bid,), tile=y_tile)


_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]


_tuner = CutileAutotuner(mul2_kernel)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n_elements,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
              kernel, (x, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
