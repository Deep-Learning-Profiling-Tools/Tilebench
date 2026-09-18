```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _quantize_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    y_tile = ct.astype(x_tile, np.float16)
    ct.store(output, index=(bid,), tile=y_tile)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    output = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 8192
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _quantize_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted to iter 3's decorator-based occupancy hint (avoiding the broken `with_hints` API), and tried a larger tile (TILE=8192, occupancy=4) to push more elements per CTA for this bandwidth-bound cast.
