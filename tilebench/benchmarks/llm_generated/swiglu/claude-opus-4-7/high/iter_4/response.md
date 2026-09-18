Looking at the trajectory: iter 1 with TILE=4096, occupancy=4 got 59.2%. Iter 3 with TILE=8192 regressed. The fp16/bf16 cases are bandwidth-bound at ~31%. Let me try smaller tiles with higher occupancy to improve latency hiding.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _swiglu_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    y_tile = ct.load(y, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x_tile, np.float32)
    yf = ct.astype(y_tile, np.float32)
    silu = xf / (1.0 + ct.exp(-xf))
    out = ct.astype(silu * yf, x.dtype)
    ct.store(output, index=(bid,), tile=out)


def run(x, y):
    output = torch.empty_like(x)
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    out_flat = output.view(-1)
    n_elements = x_flat.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _swiglu_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x_flat, y_flat, out_flat, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverting to the iter-1 baseline approach but with a smaller tile (2048) and higher occupancy (8) to increase parallelism and better hide DRAM latency for the bandwidth-bound fp16/bf16 cases.
