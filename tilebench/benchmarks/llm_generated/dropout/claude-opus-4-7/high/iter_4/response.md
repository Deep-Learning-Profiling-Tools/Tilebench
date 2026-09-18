Looking at iter 3's per-dtype breakdown:
- fp32: 93.6% roofline (TILE=8192 great for fp32 — 32KB per tile saturates bandwidth)
- fp16/bf16: only 29.9% (TILE=8192 with 2-byte dtype = only 16KB per tile, and occupancy=2 doesn't have enough CTAs in flight to saturate)

The issue is that fp16/bf16 with low occupancy doesn't saturate memory bandwidth. Going back to iter 1's TILE=4096 base, but trying higher occupancy and removing the unnecessary fp32 round-trip (which adds register pressure without improving bandwidth).

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
    # Stay in native dtype — avoid expensive fp32 round-trip for fp16/bf16.
    # scale is a Python float; it will be cast to tile dtype on multiply.
    y = x_tile * k_tile * scale
    ct.store(output, index=(bid,), tile=y)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = float(1.0 / (1.0 - p))
    stream = torch.cuda.current_stream()

    TILE = 4096
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

Approach: revert to iter 1's TILE=4096 baseline but bump occupancy to 8 for more in-flight CTAs (helps hide memory latency on fp16/bf16 where each tile is only 8KB), and drop the fp32 cast since the operation is just two multiplies that fp16/bf16 handle correctly within tolerance.
