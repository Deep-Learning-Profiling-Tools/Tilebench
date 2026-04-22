from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=8)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [1024, 2048, 4096, 8192]
    for occ in [4, 8, 16]
]
_last_autotune_config = None


@ct.kernel
def _sigmoid_kernel(x_ptr, y_ptr, TILE: ConstInt):
    """
    Element-wise sigmoid matching Triton's method:
      y[i] = 1 / (1 + exp(-x[i]))

    Compute in fp32 for accuracy, cast back to input dtype on store.
    ct.load with padding_mode=ZERO handles the last partial tile; the
    corresponding ct.store at OOB positions is silently dropped.
    """
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x_tile, np.float32)

    y_f32 = 1.0 / (1.0 + ct.exp(-x_f32))

    y_out = ct.astype(y_f32, y_ptr.dtype)
    ct.store(y_ptr, index=(bid,), tile=y_out)


def run(X: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """cuTile element-wise sigmoid mirroring Triton's tl.sigmoid method."""
    global _last_autotune_config
    output = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: (ct.cdiv(N, cfg.tile), 1, 1),
            kernel=_sigmoid_kernel,
            args_fn=lambda cfg: (X, output, cfg.tile),
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
        ct.launch(stream, grid, _sigmoid_kernel, (X, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
