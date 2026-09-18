Looking at the trajectory:
- Iter 5: TILE=2048, occupancy=8 → 77.5% (best)
- Iter 3: TILE=4096, occupancy=2 → 55.6%
- Iter 1: TILE=2048, occupancy=4 → 69.4%

Higher occupancy with TILE=2048 is winning. To push past 80%, I'll keep the winning shape but:
1. Add `latency=10` hint on loads to encourage maximum prefetch concurrency on the bandwidth-bound row sweep.
2. Simplify the compute: replace `where(pos, yt*(log_yt - lp), 0)` with `yt*log_safe - yt*lp` (when `yt==0`, both terms are exactly 0 because `safe=1 → log_safe=0`, and `lp` is finite per spec). Saves one `where` + one mul per tile.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _kl_kernel(log_y_pred, y_true, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    one = ct.full((1, TILE), 1.0, dtype=np.float32)
    for j in range(0, num_tiles):
        lp = ct.load(log_y_pred, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO, latency=10)
        yt = ct.load(y_true, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO, latency=10)
        # When yt==0: safe=1 → log_safe=0 → yt*log_safe=0; yt*lp=0 (lp finite).
        # Both contributions vanish naturally — no `where` needed on the term.
        safe = ct.where(yt > 0.0, yt, one)
        log_safe = ct.log(safe)
        acc = acc + yt * log_safe - yt * lp
    total = ct.sum(acc, axis=1, keepdims=False)
    ct.store(output, index=(row,), tile=total)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor):
    assert log_y_pred.shape == y_true.shape
    assert log_y_pred.is_contiguous() and y_true.is_contiguous()
    rows, cols = log_y_pred.shape
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    TILE = 2048
    grid = (rows, 1, 1)
    ct.launch(stream, grid, _kl_kernel, (log_y_pred, y_true, output, cols, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 8, "latency_hint": 10})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: keep the best-known configuration (TILE=2048, occupancy=8, one CTA per row) but reduce per-iteration work by dropping the redundant `where` on the term, and add `latency=10` load hints to maximize DRAM concurrency on this bandwidth-bound sweep.
