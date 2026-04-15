from types import SimpleNamespace

import cuda.tile as ct
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=2)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048, 4096, 8192]
    for occ in [1, 2, 4]
]
_last_autotune_config = None


@ct.kernel
def _leaky_relu_kernel(x_ptr, y_ptr, TILE: ConstInt):
    """
    Element-wise Leaky ReLU matching Triton's method:
      y[i] = x[i] if x[i] > 0 else 0.01 * x[i]

    ct.load with padding_mode=ZERO returns 0 for the last partial tile's
    OOB positions; computing leaky_relu on 0 gives 0 (harmless); the
    corresponding ct.store at OOB positions is silently dropped.
    """
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    y_tile = ct.where(x_tile > 0, x_tile, 0.01 * x_tile)
    ct.store(y_ptr, index=(bid,), tile=y_tile)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """cuTile element-wise Leaky ReLU mirroring Triton's tl.where method."""
    global _last_autotune_config
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (ct.cdiv(N, cfg.tile), 1, 1),
            kernel=_leaky_relu_kernel,
            args_fn=lambda cfg: (input, output, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {
            "tile":      result.tuned_config.tile,
            "occupancy": result.tuned_config.occupancy,
        }
    else:
        cfg = _DEFAULT_CONFIG
        grid = (ct.cdiv(N, cfg.tile), 1, 1)
        ct.launch(stream, grid, _leaky_relu_kernel, (input, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
