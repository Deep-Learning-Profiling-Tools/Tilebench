```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _interleave_kernel(A, B, output, TILE: ConstInt):
    bid = ct.bid(0)

    a = ct.load(
        A,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    b = ct.load(
        B,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    a2 = ct.expand_dims(a, 1)
    b2 = ct.expand_dims(b, 1)
    pairs = ct.cat((a2, b2), axis=1)
    out_tile = ct.reshape(pairs, (2 * TILE,))

    ct.store(
        output,
        index=(bid,),
        tile=out_tile,
        latency=1,
        allow_tma=False,
    )


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty((2 * N,), dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid, _interleave_kernel, (A, B, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
