Looking at the trajectory, the best was iter 1 at 60% with (TILE_M=8, TILE_N=512, occ=4). I've tried larger M, larger N, higher occupancy — all stayed near 60%. Let me try a fundamentally different shape: **TILE_M=1, TILE_N=4096** so each block processes part of one row with a *scalar* scale load (no broadcast), maximizing N-dim coalescing. Higher occupancy=8 to keep SMs busy since per-block work is small.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

TILE_M = 1
TILE_N = 4096
OCCUPANCY = 8


@ct.kernel(occupancy=OCCUPANCY)
def _dequant_kernel(x, scale, output,
                    ROWS: ConstInt, COLS: ConstInt,
                    TN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # Load int8 row-strip
    x_tile = ct.load(x, index=(bid_m, bid_n), shape=(1, TN),
                     padding_mode=ct.PaddingMode.ZERO)
    # Scalar scale load — no broadcast tile required
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

Approach: drop the 2D scale broadcast entirely — with TILE_M=1, each block handles a single row strip and loads the row's scale as a 0-D scalar tile (much cheaper than loading an `(8,)` tile and broadcasting via `[:, None]`). Larger TILE_N=4096 maximizes per-block N-direction coalescing on the int8 read and fp16 write streams.
