from types import SimpleNamespace

import torch
import cuda.tile as ct

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)

_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [1024, 2048, 4096]
    for occ in [4, 8, 16, 32]
]


@ct.kernel
def swiglu_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.astype(ct.load(x, index=(bid,), shape=(TILE,)), ct.float32)
    y_tile = ct.astype(ct.load(y, index=(bid,), shape=(TILE,)), ct.float32)
    sigmoid_x = 1.0 / (1.0 + ct.exp(-x_tile))
    out_tile = ct.astype(x_tile * sigmoid_x * y_tile, x.dtype)
    ct.store(output, index=(bid,), tile=out_tile)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(swiglu_kernel)


def run(x: torch.Tensor, y: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    assert x.shape == y.shape
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    output = torch.empty_like(x_flat)
    n_elements = x_flat.numel()
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n_elements,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((n_elements + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (x_flat, y_flat, output, cfg.tile),
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
              kernel, (x_flat, y_flat, output, cfg.tile))

    return output.view(x.shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
