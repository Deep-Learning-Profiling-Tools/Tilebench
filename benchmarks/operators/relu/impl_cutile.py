from types import SimpleNamespace

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
def _relu_kernel(x_ptr, output_ptr, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,))
    zero = ct.zeros((TILE,), dtype=x_tile.dtype)
    y_tile = ct.where(x_tile >= 0, x_tile, zero)
    ct.store(output_ptr, index=(bid,), tile=y_tile)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_autotune_config
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    if autotune:
        result = ct.tune.exhaustive_search(
            _SEARCH_SPACE,
            stream,
            grid_fn=lambda cfg: (ct.cdiv(n_elements, cfg.tile), 1, 1),
            kernel=_relu_kernel,
            args_fn=lambda cfg: (x, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        cfg = result.best.config
        _last_autotune_config = {
            "tile":      cfg.tile,
            "occupancy": cfg.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG

    ct.launch(stream, (ct.cdiv(n_elements, cfg.tile), 1, 1),
              _relu_kernel, (x, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
