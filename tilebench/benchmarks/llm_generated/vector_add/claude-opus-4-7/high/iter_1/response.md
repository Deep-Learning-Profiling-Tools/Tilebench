Fixing the `with_hints` error by applying occupancy as a decorator argument, and bumping tile size for better bandwidth.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _add_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    y_tile = ct.load(y, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    ct.store(output, index=(bid,), tile=x_tile + y_tile)


def run(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _add_kernel, (x, y, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Removed broken `with_hints` call; occupancy is now set via the `@ct.kernel(occupancy=4)` decorator argument. Kept TILE=4096 as a solid starting point for memory-bound 1D pointwise on B200.
