Looking at iter 6 (best so far at 77.8%) — int8 is the worst case at 51.2%. int8 with TILE=4096 means only 4KB per CTA which has poor amortization. Let me try larger tiles to improve int8 bandwidth.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _relu_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    zero = ct.full(x_tile.shape, 0, dtype=x_tile.dtype)
    y_tile = ct.where(x_tile > zero, x_tile, zero)
    ct.store(output, index=(bid,), tile=y_tile)


def run(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 8192
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _relu_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Trying TILE=8192 with occupancy=4 to improve amortization especially for int8 (8KB tiles vs 4KB), while keeping enough CTAs for good wave occupancy on B200's 148 SMs.
