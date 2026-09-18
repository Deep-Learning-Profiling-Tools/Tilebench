Looking at the trajectory:
- iter 1: TILE=8192, occ=4 → 71.4%
- iter 3 (best): TILE=4096, occ=8 → 77.7%
- iter 5: TILE=4096, occ=4 → 77.6%
- iter 6: TILE=4096, occ=16 → 74.8%

TILE=4096 is the sweet spot. occupancy=8 was best. Let me try TILE=2048 with occupancy=8 to give the scheduler more parallelism and a `latency` hint on the load/store for better DRAM pipelining.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _mul2_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO,
                     latency=10)
    y_tile = x_tile * 2
    ct.store(output, index=(bid,), tile=y_tile, latency=10)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _mul2_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "latency": 10})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: Reverted to best-known TILE/occupancy point (4096/8 was best, trying 2048/8 with maximum DRAM latency hints to get more in-flight memory operations) and added explicit `latency=10` hints on load/store so the scheduler issues more async copies for this purely bandwidth-bound kernel.
