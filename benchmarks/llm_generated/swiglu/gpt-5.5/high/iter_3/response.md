```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _swiglu_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    y_tile = ct.load(
        y,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    xf = ct.astype(x_tile, np.float32)
    yf = ct.astype(y_tile, np.float32)

    one = ct.full((TILE,), 1.0, dtype=np.float32)

    # sigmoid(x) = 1 / (1 + exp(-x)) = 1 / (1 + exp2(-x * log2(e))).
    # exp2 + approximate reciprocal is faster than the direct exp/div path,
    # while keeping the accumulation path in fp32 for verification stability.
    exp_term = ct.exp2(xf * -1.4426950408889634, flush_to_zero=True)
    sig = ct.truediv(
        one,
        one + exp_term,
        rounding_mode=RMd.APPROX,
        flush_to_zero=True,
    )

    out = ct.astype((xf * sig) * yf, x.dtype)

    ct.store(
        output,
        index=(bid,),
        tile=out,
        latency=1,
        allow_tma=False,
    )


def run(x, y):
    output = torch.empty_like(x)

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    out_flat = output.reshape(-1)

    n_elements = x_flat.numel()
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, _swiglu_kernel, (x_flat, y_flat, out_flat, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "occupancy": occupancy,
            "EXP2_SIGMOID": 1,
            "APPROX_RCP": 1,
            "FP32_MATH": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
