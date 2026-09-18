Going back to iter 3's working code (TILE=4096, occupancy=8) and trying a small change: adding explicit high-latency hints on load/store to encourage better memory pipelining without breaking the API.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
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

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _mul2_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Kept iter-3 code structure (decorator-based occupancy hint, no `with_hints` API call). Lowered occupancy to 4 to give each CTA more registers/shared memory bandwidth for the pure memory-bound copy, and added `latency=10` hints to encourage better software pipelining of the DRAM accesses.
