Looking at the trajectory: fp32 gets 95.6% but fp16/bf16 only 48-49% with TILE=4096. The bottleneck for low-byte dtypes is likely too-small per-tile work amortizing poorly. Doubling TILE to 8192 should help fp16/bf16 while staying within elementwise upper bounds.

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _dropout_kernel(x, x_keep, output, scale: float, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    k_tile = ct.load(x_keep, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x_tile, np.float32)
    k_f32 = ct.astype(k_tile, np.float32)
    y_f32 = x_f32 * scale * k_f32
    y = ct.astype(y_f32, x.dtype)
    ct.store(output, index=(bid,), tile=y)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = float(1.0 / (1.0 - p))
    stream = torch.cuda.current_stream()

    TILE = 8192
    occupancy = 4
    kernel = _dropout_kernel.with_hints(occupancy=occupancy)
    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    ct.launch(stream, grid, kernel, (x, x_keep, output, scale, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Bumped TILE from 4096 → 8192 to give fp16/bf16 enough per-tile bytes to saturate memory pipelines (was 24KB now 48KB per tile across the 3 streams), while keeping occupancy=4 since fp32 was already achieving 95% roofline.
