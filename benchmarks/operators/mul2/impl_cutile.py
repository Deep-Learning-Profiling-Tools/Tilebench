import math
from types import SimpleNamespace

import torch
import cuda.tile as ct

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_config: dict | None = None


@ct.kernel
def mul2_kernel(x_ptr, output_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,))
    y_tile = x_tile * 2
    ct.store(output_ptr, index=(bid,), tile=y_tile)


# Search space for B200 (sm_100, 148 SMs, HBM3e ~8 TB/s).
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]


def run(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    global _last_config
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    if ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (math.ceil(n_elements / cfg.tile), 1, 1),
            kernel=mul2_kernel,
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_config = {
            "tile":      result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        tile = block_size
        grid = (math.ceil(n_elements / tile), 1, 1)
        ct.launch(stream, grid, mul2_kernel, (x, output, tile))

    return output


def get_last_config() -> dict | None:
    return _last_config
