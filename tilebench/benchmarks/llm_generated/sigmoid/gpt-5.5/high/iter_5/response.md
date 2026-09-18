```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _sigmoid_kernel(X, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        X,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    x_fp32 = ct.astype(x_tile, np.float32)

    # Mathematically equivalent to exp(-x), but exp2 maps more directly to
    # the hardware SFU path. Keep fp32 math before the final cast to match
    # torch.sigmoid(X.float()).to(X.dtype) within normal verification tolerances.
    e_fp32 = ct.exp2(x_fp32 * -1.4426950408889634, flush_to_zero=True)
    y_fp32 = ct.truediv(
        1.0,
        1.0 + e_fp32,
        rounding_mode=RMd.APPROX,
        flush_to_zero=True,
    )

    y_tile = ct.astype(y_fp32, X.dtype)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        latency=1,
        allow_tma=False,
    )


def run(X: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(X)
    n_elements = X.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _sigmoid_kernel, (X, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
        "fast_math": "exp2_approx_rcp",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
