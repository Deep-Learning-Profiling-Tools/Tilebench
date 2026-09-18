Looking at the trajectory:
- iter 1: TILE=8192, occ=4 → 71.4%
- iter 3: TILE=4096, occ=8 → 77.7% (best)

int8 is the worst (65%) — likely because per-tile bytes are smallest. Let me try TILE=8192 with occupancy=8 to give int8 more bytes per tile while keeping high SM occupancy, plus a latency hint for streaming.

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

    TILE = 8192
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _mul2_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: bumped TILE to 8192 to give the worst-performing dtype (int8) more bytes per tile (8 KB vs 4 KB), kept occupancy=8 (vs iter 1's occ=4 which underperformed), and added `latency=10` streaming hints on both load and store to bias the compiler toward maximizing in-flight DRAM traffic.
