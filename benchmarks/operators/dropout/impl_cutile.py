from types import SimpleNamespace

import numpy as np
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def _dropout_kernel(x, x_keep, output, scale, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile      = ct.astype(ct.load(x,      index=(bid,), shape=(TILE,)), np.float32)
    x_keep_tile = ct.astype(ct.load(x_keep, index=(bid,), shape=(TILE,)), np.float32)
    out_tile = ct.astype(x_keep_tile * x_tile * scale, x.dtype)
    ct.store(output, index=(bid,), tile=out_tile)


def run(x: torch.Tensor, x_keep: torch.Tensor, p: float,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = 1.0 / (1.0 - p)
    stream = torch.cuda.current_stream()

    if autotune:
        result = ct.tune.exhaustive_search(
            _SEARCH_SPACE,
            stream,
            grid_fn=lambda cfg: ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=_dropout_kernel,
            args_fn=lambda cfg: (x, x_keep, output, scale, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        cfg = result.best.config
        _last_autotune_config = {
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG

    ct.launch(stream, ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
              _dropout_kernel, (x, x_keep, output, scale, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
