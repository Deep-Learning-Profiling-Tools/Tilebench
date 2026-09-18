```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8, opt_level=3)
def _dropout_kernel(x, x_keep, output, scale, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    keep_tile = ct.load(
        x_keep,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )

    scaled = x_tile * scale
    out_tile = ct.where(keep_tile != 0, scaled, 0.0)

    ct.store(output, index=(bid,), tile=out_tile, allow_tma=False)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    scale = 1.0 / (1.0 - float(p))
    grid = (ct.cdiv(n_elements, TILE), 1, 1)

    ct.launch(stream, grid, _dropout_kernel, (x, x_keep, output, scale, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
