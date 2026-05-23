Looking at the trajectory, iter 8 (TILE_M=1, TILE_N=2048, occupancy=16) achieved the best score at 74.4%. Smaller TILE_N with higher occupancy seems to help — let me try pushing occupancy further to 32, which is the max allowed.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

TILE_M = 1
TILE_N = 2048
OCCUPANCY = 32


@ct.kernel(occupancy=OCCUPANCY)
def _dequant_kernel(x, scale, output,
                    ROWS: ConstInt, COLS: ConstInt,
                    TN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(1, TN),
                     padding_mode=ct.PaddingMode.ZERO,
                     latency=10)
    s = ct.load(scale, index=(bid_m,), shape=(), latency=1)

    x_f = ct.astype(x_tile, np.float32)
    y = x_f * (s * np.float32(1.0 / 127.0))
    y = ct.astype(y, output.dtype)

    ct.store(output, index=(bid_m, bid_n), tile=y)


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows, cols = x.shape
    output = torch.empty((rows, cols), dtype=torch.float16, device=x.device)
    stream = torch.cuda.current_stream()

    grid = (rows, ct.cdiv(cols, TILE_N), 1)
    ct.launch(stream, grid, _dequant_kernel,
              (x, state_x, output, rows, cols, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_M": TILE_M, "TILE_N": TILE_N, "occupancy": OCCUPANCY})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Keeping the iter-8 winning tile (TILE_M=1, TILE_N=2048) but bumping occupancy from 16 → 32 to expose more parallelism and better hide the int8 read / fp16 write latency on B200.
