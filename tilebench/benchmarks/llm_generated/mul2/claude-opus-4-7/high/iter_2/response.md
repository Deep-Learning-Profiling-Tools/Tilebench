Looking at iter 1: fp32 hits 91.7% but int8/fp16/bf16 underperform (58-68%). This suggests smaller-dtype tiles aren't generating enough memory-level parallelism — at TILE=8192 elements, int8 only moves 16KB per block (vs 64KB for fp32). Let me try a larger tile to give every block more bandwidth work, and keep good occupancy.

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
                     padding_mode=ct.PaddingMode.ZERO)
    y_tile = x_tile * 2
    ct.store(output, index=(bid,), tile=y_tile)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 16384
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _mul2_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Bumped TILE from 8192 to 16384 to double the per-block bytes (especially helps int8/fp16 where each element is small), keeping occupancy=4 for SM-level parallelism.
