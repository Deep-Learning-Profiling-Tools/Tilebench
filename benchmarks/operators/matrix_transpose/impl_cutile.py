from types import SimpleNamespace

import torch
import cuda.tile as ct

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile=32, occupancy=2)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [8, 16, 32, 64]
    for occ in [1, 2, 4]
]


@ct.kernel
def _transpose_kernel(x, output, TILE: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(TILE, TILE))
    ct.store(output, index=(bid_n, bid_m), tile=ct.transpose(x_tile))


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    m, n = x.shape
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: ((m + cfg.tile - 1) // cfg.tile, (n + cfg.tile - 1) // cfg.tile, 1),
            kernel=_transpose_kernel,
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile":      result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        grid = ((m + cfg.tile - 1) // cfg.tile, (n + cfg.tile - 1) // cfg.tile, 1)
        ct.launch(stream, grid, _transpose_kernel, (x, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
