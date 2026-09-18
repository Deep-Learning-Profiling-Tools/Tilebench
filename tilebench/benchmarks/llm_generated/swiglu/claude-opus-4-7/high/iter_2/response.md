Looking at the feedback: fp16/bf16 are at 38.9% (bandwidth-bound) while fp32 hits 104%. The fp32 case effectively moves 2× more bytes per thread for the same TILE size, suggesting the fp16/bf16 cases need a larger per-CTA workload to amortize launch overhead and saturate memory bandwidth. I'll try TILE=8192 with higher occupancy.

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

    TILE = 8192
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

Approach: Doubled TILE to 8192 and bumped occupancy to 8 to provide more in-flight memory transactions per SM, targeting the bandwidth-bound fp16/bf16 cases that stalled at ~3TB/s out of B200's ~8TB/s.
