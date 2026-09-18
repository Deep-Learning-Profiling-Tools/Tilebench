Looking at iter 1: fp32 hit 98.8% (great), but fp16/bf16 only hit ~48%. This suggests for smaller dtypes the tile is too small to saturate bandwidth (per-tile launch/instruction overhead dominates). Doubling the tile to 8192 should help fp16/bf16 while keeping fp32 within bounds.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _sigmoid_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x_tile, np.float32)
    y = 1.0 / (1.0 + ct.exp(-xf))
    y_out = ct.astype(y, x.dtype)
    ct.store(output, index=(bid,), tile=y_out)


def run(X: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(X)
    n_elements = X.numel()
    stream = torch.cuda.current_stream()

    TILE = 8192
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _sigmoid_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (X, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Increased TILE from 4096 → 8192 to amortize per-tile overhead for fp16/bf16, which underperformed at the smaller tile size.
