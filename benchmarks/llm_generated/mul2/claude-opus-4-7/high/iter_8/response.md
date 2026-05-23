Looking at the trajectory:
- iter 3 (best): TILE=4096, occupancy=8 → 77.7%
- iter 1: TILE=8192, occ=4 → 71.4%
- iter 5: TILE=4096, occ=4 → 77.6%
- iter 6: TILE=4096, occ=16 → 74.8%

The sweet spot is TILE=4096 with occupancy around 8. Let me try adding `latency=1` hints on loads/stores (since this is purely memory-bound, low latency hint should help DRAM scheduling) while keeping the working decorator syntax.

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
                     latency=1)
    y_tile = x_tile * 2
    ct.store(output, index=(bid,), tile=y_tile, latency=1)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _mul2_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "latency": 1})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverting to iter 3's working decorator syntax (`@ct.kernel(occupancy=8)`) and adding `latency=1` hints to both load and store — since mul2 is purely DRAM-bandwidth-bound, signaling low-latency intent to the scheduler may improve DRAM traffic overlap without changing the tile size that was already optimal.
