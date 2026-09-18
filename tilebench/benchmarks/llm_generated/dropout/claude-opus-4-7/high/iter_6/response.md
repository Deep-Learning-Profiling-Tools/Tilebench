Looking at the trajectory:
- Best was iter 5 (TILE=2048, occupancy=8) at 74.2%
- fp32 is at 95% (great), but fp16/bf16 are stuck at ~63%
- The kernel currently casts to fp32 unnecessarily — for fp16/bf16 this adds conversion overhead, but for fp32 it's free

The fix: skip the fp32 round-trip and multiply natively in the input dtype. Since `x_keep` values are 0/1 and `scale` is a benign multiplier, native fp16/bf16 arithmetic is fine.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _dropout_kernel(x, x_keep, output, scale: float, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    k_tile = ct.load(x_keep, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    # Multiply natively in input dtype — avoids fp16/bf16 -> fp32 -> fp16/bf16
    # conversion overhead. scale (Python float) auto-promotes to tile dtype.
    y = x_tile * k_tile * scale
    ct.store(output, index=(bid,), tile=y)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = float(1.0 / (1.0 - p))
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8
    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _dropout_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, x_keep, output, scale, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Removed the explicit `astype(np.float32)` conversions which were costing memory bandwidth efficiency on fp16/bf16. Kept TILE=2048, occupancy=8 (best verified config from iter 5). The scale multiply now happens in native dtype, eliminating two cast passes per element on the half-precision cases where they were dragging down throughput.
