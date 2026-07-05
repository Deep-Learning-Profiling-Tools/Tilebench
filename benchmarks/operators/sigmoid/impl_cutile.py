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
def sigmoid_kernel(x_ptr, y_ptr, TILE: ConstInt):
    """
    Element-wise sigmoid matching Triton's method:
      y[i] = 1 / (1 + exp(-x[i]))

    Compute in fp32 for accuracy, cast back to input dtype on store.
    ct.load with padding_mode=ZERO handles the last partial tile; the
    corresponding ct.store at OOB positions is silently dropped.
    """
    bid = ct.bid(0)
    x_tile = ct.load(x_ptr, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x_tile, ct.float32)

    y_f32 = 1.0 / (1.0 + ct.exp(-x_f32))

    y_out = ct.astype(y_f32, y_ptr.dtype)
    ct.store(y_ptr, index=(bid,), tile=y_out)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
_tuner = CutileAutotuner(sigmoid_kernel)


def run(X: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """cuTile element-wise sigmoid mirroring Triton's tl.sigmoid method."""
    output = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (ct.cdiv(N, cfg.tile), 1, 1),
            args_fn=lambda cfg: (X, output, cfg.tile),
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
    ct.launch(stream, grid, kernel, (X, output, cfg.tile))

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
