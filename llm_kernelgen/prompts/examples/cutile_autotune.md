# Train Example: cuTile with CutileAutotuner (dropout)

**Operator:** `dropout`
**Backend:** cuTile
**Pattern:** Using `CutileAutotuner` for occupancy tuning

```python
import math
from types import SimpleNamespace

import torch
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_config: dict | None = None

_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [1, 2, 4]]

@ct.kernel
def _dropout_kernel(X, X_Keep, Out, N: ConstInt, TILE: ConstInt):
    bid  = ct.bid(0)
    x    = ct.load(X,      index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    keep = ct.load(X_Keep, index=(bid,), shape=(TILE,), padding_mode=ct.PaddingMode.ZERO)
    out  = x * keep
    ct.store(Out, index=(bid,), tile=out)


_tuner = CutileAutotuner(_dropout_kernel)


def run(x: torch.Tensor, x_keep: torch.Tensor, p: float,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_config
    n      = x.numel()
    out    = torch.empty_like(x)
    TILE   = block_size
    stream = torch.cuda.current_stream().cuda_stream
    grid   = (math.ceil(n / TILE),)

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(n, TILE),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda c: grid,
            args_fn=lambda c: (x, x_keep, out, n, TILE),
            hints_fn=lambda c: {"occupancy": c.occupancy},
        )
        _last_config = {"occupancy": cfg.occupancy}
    else:
        cfg = SimpleNamespace(occupancy=1)
        _last_config = None

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(stream, grid, kernel, (x, x_keep, out, n, TILE))
    return out


def get_last_config() -> dict | None:
    return _last_config
```
