from types import SimpleNamespace

import numpy as np
import torch
import cuda.tile as ct

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]


@ct.kernel
def _swiglu_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.astype(ct.load(x, index=(bid,), shape=(TILE,)), np.float32)
    y_tile = ct.astype(ct.load(y, index=(bid,), shape=(TILE,)), np.float32)
    sigmoid_x = 1.0 / (1.0 + ct.exp(-x_tile))
    out_tile = ct.astype(x_tile * sigmoid_x * y_tile, x.dtype)
    ct.store(output, index=(bid,), tile=out_tile)


def run(x: torch.Tensor, y: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    assert x.shape == y.shape
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    output = torch.empty_like(x_flat)
    n_elements = x_flat.numel()
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
            kernel=_swiglu_kernel,
            args_fn=lambda cfg: (x_flat, y_flat, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile":      result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        ct.launch(stream, ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
                  _swiglu_kernel, (x_flat, y_flat, output, cfg.tile))

    return output.view(x.shape)


def get_last_config() -> dict | None:
    return _last_autotune_config
