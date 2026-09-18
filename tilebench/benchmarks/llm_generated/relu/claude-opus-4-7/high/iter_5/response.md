Going back to iter 1's approach (TILE=4096, occupancy=4) which was the best, and bumping occupancy higher to keep more memory transactions in flight (bandwidth-bound op).

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _relu_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO,
                     latency=10)
    zero = ct.full(x_tile.shape, 0, dtype=x_tile.dtype)
    y_tile = ct.maximum(x_tile, zero)
    ct.store(output, index=(bid,), tile=y_tile)


def run(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _relu_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted to iter 1's simple single-tile-per-CTA approach but increased occupancy from 4→8 and added `latency=10` hint to maximize in-flight DRAM transactions for this bandwidth-bound op.
