Looking at the trajectory, iter 1 (TILE=2048, occupancy=4) was the best at 63.4%. The big algorithmic win I haven't tried is **caching x tiles between pass 1 and pass 2**, eliminating the duplicate x reads. The standard 2-pass LayerNorm reads x twice (3n bytes total), but if we save the loaded fp32 tiles in a Python list during the unrolled loop, we read x only once (2n bytes effective). For K=10240, TILE=2048 → 5 cached tiles = 40KB per CTA, which fits comfortably with occupancy=2.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _ln_kernel(x, w, b, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    inv_N = 1.0 / N

    # Pass 1: load x once, cache fp32 tiles, accumulate scalar partial sums.
    sum_x = 0.0
    sum_xx = 0.0
    saved = []
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    latency=10),
            np.float32)
        saved.append(xj)
        sum_x = sum_x + ct.sum(xj)
        sum_xx = sum_xx + ct.sum(xj * xj)

    mean = sum_x * inv_N
    var = sum_xx * inv_N - mean * mean
    rstd = ct.rsqrt(var + eps)

    # Pass 2: reuse cached x tiles; only w, b, and output touch DRAM.
    for j in range(0, num_tiles):
        wj = ct.astype(
            ct.load(w, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32)
        bj = ct.astype(
            ct.load(b, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32)
        yj = (saved[j] - mean) * rstd * wj.reshape((1, TILE)) + bj.reshape((1, TILE))
        ct.store(out, index=(row, j), tile=ct.astype(yj, x.dtype))


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5):
    out = torch.empty_like(x)
    K = x.shape[-1]
    rows = x.numel() // K
    x2d = x.contiguous().view(rows, K)
    out2d = out.view(rows, K)
    stream = torch.cuda.current_stream()

    TILE = 2048

    ct.launch(stream, (rows, 1, 1), _ln_kernel,
              (x2d, weight, bias, out2d, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 2, "cache_x_tiles": True})
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: cache the fp32-upcast x tiles in a Python list during the compile-time unrolled pass-1 loop, then reuse them in pass 2 — cutting DRAM x-reads in half and dropping per-CTA traffic from ~3n to ~2n bytes. Reduced occupancy to 2 to give each CTA enough registers/shmem for the 5 cached tiles at TILE=2048.
