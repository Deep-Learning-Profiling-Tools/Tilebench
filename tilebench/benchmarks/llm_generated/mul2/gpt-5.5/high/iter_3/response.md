```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _mul2_kernel(
    x,
    output,
    TILE: ConstInt,
    TILES_PER_CTA: ConstInt,
    PAD_TAIL: ConstBool,
):
    bid = ct.bid(0)
    base = bid * TILES_PER_CTA

    for t in range(0, TILES_PER_CTA):
        tile_idx = base + t

        if PAD_TAIL:
            x_tile = ct.load(
                x,
                index=(tile_idx,),
                shape=(TILE,),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
        else:
            x_tile = ct.load(
                x,
                index=(tile_idx,),
                shape=(TILE,),
                latency=1,
                allow_tma=False,
            )

        y_tile = x_tile + x_tile

        ct.store(
            output,
            index=(tile_idx,),
            tile=y_tile,
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    TILES_PER_CTA = 2
    occupancy = 8
    elements_per_cta = TILE * TILES_PER_CTA
    PAD_TAIL = (n_elements % elements_per_cta) != 0

    grid = (ct.cdiv(n_elements, elements_per_cta), 1, 1)
    ct.launch(stream, grid, _mul2_kernel, (x, output, TILE, TILES_PER_CTA, PAD_TAIL))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "TILES_PER_CTA": TILES_PER_CTA,
            "occupancy": occupancy,
            "PAD_TAIL": PAD_TAIL,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
