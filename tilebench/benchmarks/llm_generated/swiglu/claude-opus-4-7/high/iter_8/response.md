Looking at the trajectory: iter 5 (TILE=4096, occ=8) gave the best score at 75%. The fp16/bf16 cases are bandwidth-bound at ~55%. Let me try increasing the tile size with the right occupancy to improve memory throughput — TILE=8192 with occupancy=8 (iter 3 used occ=4 and got 53.9%, but with higher occupancy may be better).

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

Trying TILE=2048 with occupancy=8 — smaller tiles increase the number of CTAs (more parallelism, better SM utilization across the large N=20480 case) while keeping per-CTA register/shmem pressure low to maintain occupancy.
