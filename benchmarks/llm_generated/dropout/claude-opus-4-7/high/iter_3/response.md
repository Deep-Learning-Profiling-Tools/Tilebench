Reverting to iter 1's working approach (decorator-style occupancy) and trying a larger TILE=8192 for better memory throughput.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _dropout_kernel(x, x_keep, output, scale: float, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    k_tile = ct.load(x_keep, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x_tile, np.float32)
    k_f32 = ct.astype(k_tile, np.float32)
    y_f32 = x_f32 * scale * k_f32
    y = ct.astype(y_f32, x.dtype)
    ct.store(output, index=(bid,), tile=y)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = float(1.0 / (1.0 - p))
    stream = torch.cuda.current_stream()

    TILE = 8192
    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _dropout_kernel, (x, x_keep, output, scale, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Kept the working decorator-style `@ct.kernel(occupancy=2)` pattern from iter 1, doubled TILE to 8192 and lowered occupancy to 2 to amortize launch overhead per CTA while keeping enough SMs busy for the 20M-element case.
