Looking at the trajectory:
- Best so far: iter 5 (TILE_M=1, TILE_N=4096, occ=8) at 67.2%
- iter 6 regressed to 50% (TILE_N=8192 too large)

Let me try `TILE_M=2, TILE_N=4096, occupancy=8` — this halves the grid (4096 vs 8192 CTAs) and amortizes per-CTA launch overhead by processing 2 rows together while keeping the per-tile working set modest (8KB int8 in, 16KB fp16 out per CTA).

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

TILE_M = 2
TILE_N = 4096
OCCUPANCY = 8


@ct.kernel(occupancy=OCCUPANCY)
def _dequant_kernel(x, scale, output,
                    ROWS: ConstInt, COLS: ConstInt,
                    TM: ConstInt, TN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Load TM rows × TN cols int8 strip
    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(TM, TN),
                     padding_mode=ct.PaddingMode.ZERO)
    # Load TM scales (column vector via broadcasting)
    s = ct.load(scale, index=(bid_m,), shape=(TM,),
                padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x_tile, np.float32)
    y = x_f * s[:, None] * np.float32(1.0 / 127.0)
    y = ct.astype(y, output.dtype)

    ct.store(output, index=(bid_m, bid_n), tile=y)


def run(x: torch.Tensor, state_x: torch.Tensor, **kwargs):
    rows, cols = x.shape
    output = torch.empty((rows, cols), dtype=torch.float16, device=x.device)
    stream = torch.cuda.current_stream()

    grid = (ct.cdiv(rows, TILE_M), ct.cdiv(cols, TILE_N), 1)
    ct.launch(stream, grid, _dequant_kernel,
              (x, state_x, output, rows, cols, TILE_M, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_M": TILE_M, "TILE_N": TILE_N, "occupancy": OCCUPANCY})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted from iter 6's regression back to iter 5's TILE_N=4096 + occ=8 baseline, but bumped TILE_M from 1→2 to halve the grid count and amortize launch/scale-load overhead.
