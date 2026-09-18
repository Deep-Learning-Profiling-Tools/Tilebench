Going to try TILE_N=8192 (full row in single block for the largest case) — this halves the number of blocks vs iter 5 and reads each row's scale only once per row.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

TILE_M = 1
TILE_N = 8192
OCCUPANCY = 4


@ct.kernel(occupancy=OCCUPANCY)
def _dequant_kernel(x, scale, output,
                    ROWS: ConstInt, COLS: ConstInt,
                    TN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(1, TN),
                     padding_mode=ct.PaddingMode.ZERO)
    s = ct.load(scale, index=(bid_m,), shape=())

    x_f = ct.astype(x_tile, np.float32)
    y = x_f * s * np.float32(1.0 / 127.0)
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

Approach: single block covers the full row (TILE_N=8192 ≥ max cols), so each row's scale is loaded exactly once per block instead of multiple times across column-blocks; reduced occupancy to 4 since each block now does 2× more work.
