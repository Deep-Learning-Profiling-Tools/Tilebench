Looking at the error: `with_hints` is not a valid method. I need to set occupancy via the `@ct.kernel(occupancy=...)` decorator directly.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

BM = 128
BN = 128
OCCUPANCY = 4


@ct.kernel(occupancy=OCCUPANCY)
def _transpose_kernel(x, output, M: ConstInt, N: ConstInt,
                      BM_C: ConstInt, BN_C: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    tile = ct.load(x, index=(bid_m, bid_n), shape=(BM_C, BN_C),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_t = ct.transpose(tile)
    ct.store(output, index=(bid_n, bid_m), tile=tile_t)


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    M, N = x.shape
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)
    stream = torch.cuda.current_stream()

    grid = (ct.cdiv(M, BM), ct.cdiv(N, BN), 1)
    ct.launch(stream, grid, _transpose_kernel, (x, output, M, N, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BM": BM, "BN": BN, "occupancy": OCCUPANCY})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Fixed the `with_hints` error by moving the occupancy hint to the `@ct.kernel(occupancy=...)` decorator. Also bumped tile size from 64×64 to 128×128 for better bandwidth utilization on B200.
