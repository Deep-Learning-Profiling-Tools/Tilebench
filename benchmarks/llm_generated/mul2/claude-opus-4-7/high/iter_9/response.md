Looking at the trajectory: iter 3 (TILE=4096, occupancy=8, no hints) is the best at 77.7%. Adding `latency=1` regressed slightly. Let me try a high latency hint (10) on the load to encourage software pipelining of memory transactions, since this is purely bandwidth-bound.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _mul2_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO,
                     latency=10)
    y_tile = x_tile * 2
    ct.store(output, index=(bid,), tile=y_tile)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _mul2_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "load_latency": 10})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: Keep the iter-3 best config (TILE=4096, occupancy=8) but add a high `latency=10` hint on the load to encourage the compiler to schedule the DRAM read further ahead of the store, enabling deeper memory-pipeline overlap on this bandwidth-bound op.
