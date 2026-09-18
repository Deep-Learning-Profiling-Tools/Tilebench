```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
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

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _quantize_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Switched to `TILE=2048, occupancy=8` — for a pure bandwidth-bound dtype cast, higher CTA occupancy (more concurrent in-flight loads/stores per SM) typically wins over larger per-tile work, since each tile already coalesces fully and saturates the LSU at the smaller size.
