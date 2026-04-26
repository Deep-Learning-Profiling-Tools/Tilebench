from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_DEFAULT_CONFIG = SimpleNamespace(tile=64, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [32, 64, 128]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def _transpose_kernel(x, output, TILE: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(TILE, TILE))
    ct.store(output, index=(bid_n, bid_m), tile=ct.transpose(x_tile))


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(_transpose_kernel)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    m, n = x.shape
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(m, n),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((m + cfg.tile - 1) // cfg.tile, (n + cfg.tile - 1) // cfg.tile, 1),
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config = {
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG

    grid = ((m + cfg.tile - 1) // cfg.tile, (n + cfg.tile - 1) // cfg.tile, 1)
    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel, (x, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
