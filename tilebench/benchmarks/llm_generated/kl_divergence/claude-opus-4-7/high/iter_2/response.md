Looking at iter 1, TILE=2048 with 8 iterations per row achieved 69.4% on the largest case (4096×16384, fp32). To push toward 80%, I'll:
1. **Increase tile size to 8192** — only 2 iterations per row → larger TMA bursts, less loop overhead.
2. **Drop the `where`-based safe-log construction** in favor of `ct.maximum(yt, tiny)` so `log` always sees a finite input, and OOB padded zeros naturally contribute 0 via `yt * (...)`.
3. **occupancy=2** — fewer active CTAs but more bytes-per-block keeps HBM streams saturated.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

# Small positive constant so log(max(yt, _TINY)) is finite when yt==0;
# yt==0 still contributes 0 via the outer multiply (0 * finite == 0).
_TINY = 1.0e-30


@ct.kernel
def _kl_kernel(log_y_pred, y_true, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        lp = ct.load(log_y_pred, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        yt = ct.load(y_true, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO)
        # Clamp y_true so log() is always finite; outer multiply by the
        # original yt produces 0 wherever yt == 0 (including OOB pads).
        yt_safe = ct.maximum(yt, ct.full((1, TILE), _TINY, dtype=np.float32))
        log_yt = ct.log(yt_safe)
        acc = acc + yt * (log_yt - lp)
    total = ct.sum(acc, axis=1, keepdims=False)
    ct.store(output, index=(row,), tile=total)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor):
    assert log_y_pred.shape == y_true.shape
    assert log_y_pred.is_contiguous() and y_true.is_contiguous()
    rows, cols = log_y_pred.shape
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    TILE = 8192
    occupancy = 2

    grid = (rows, 1, 1)
    kernel = _kl_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (log_y_pred, y_true, output, cols, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Increasing TILE to 8192 (2 iterations across the 16384-col reduction) and using `ct.maximum`+`log` (instead of the masked `where`+`log` pair) should improve HBM utilization and trim per-iteration overhead.
