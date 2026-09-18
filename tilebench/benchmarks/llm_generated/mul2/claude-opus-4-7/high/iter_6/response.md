Reverting to the iter-3 baseline (TILE=4096, occupancy=8) which is the best verify-clean config, and bumping occupancy to 16 to allow more in-flight memory operations per SM — the kernel is bandwidth-bound and int8 (the worst case at 64.7%) particularly benefits from more concurrent CTAs hiding DRAM latency on small per-CTA payloads.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=16)
def _mul2_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    y_tile = x_tile * 2
    ct.store(output, index=(bid,), tile=y_tile)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 16

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _mul2_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
