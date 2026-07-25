from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [1024, 2048, 4096, 8192]
    for occ in [4, 8, 16]
]
_last_autotune_config: dict = {}


@ct.kernel
def leaky_relu_kernel(x_ptr, y_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    y_tile = ct.where(x_tile > 0, x_tile, 0.01 * x_tile)
    ct.store(y_ptr, index=(bid,), tile=y_tile)


_tuner = CutileAutotuner(leaky_relu_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(N, cfg.tile), 1, 1),
            args_fn=lambda cfg: (input, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    grid = (ct.cdiv(N, cfg.tile), 1, 1)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel, (input, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
