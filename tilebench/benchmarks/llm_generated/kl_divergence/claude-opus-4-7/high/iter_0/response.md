```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _kl_kernel(log_y_pred_ptr, y_true_ptr, out_ptr,
               n_cols, stride_row,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = row * stride_row
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        lp = tl.load(log_y_pred_ptr + base + cols, mask=mask, other=0.0)
        yt = tl.load(y_true_ptr + base + cols, mask=mask, other=0.0)
        pos = yt > 0.0
        log_yt = tl.log(tl.where(pos, yt, 1.0))
        term = tl.where(pos, yt * (log_yt - lp), 0.0)
        acc += term
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor):
    assert log_y_pred.shape == y_true.shape
    assert log_y_pred.is_contiguous() and y_true.is_contiguous()
    rows, cols = log_y_pred.shape
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)

    BLOCK_N = 2048
    num_warps = 8
    num_stages = 4

    grid = (rows,)
    _kl_kernel[grid](
        log_y_pred, y_true, output,
        cols, log_y_pred.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N, "num_warps": num_warps, "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


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
        pos = yt > 0.0
        safe = ct.where(pos, yt, ct.full((1, TILE), 1.0, dtype=np.float32))
        log_yt = ct.log(safe)
        term = ct.where(pos, yt * (log_yt - lp),
                        ct.full((1, TILE), 0.0, dtype=np.float32))
        acc = acc + term
    total = ct.sum(acc, axis=1, keepdims=False)
    ct.store(output, index=(row,), tile=total)


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor):
    assert log_y_pred.shape == y_true.shape
    assert log_y_pred.is_contiguous() and y_true.is_contiguous()
    rows, cols = log_y_pred.shape
    output = torch.empty((rows,), device=log_y_pred.device, dtype=torch.float32)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 4

    grid = (rows, 1, 1)
    kernel = _kl_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (log_y_pred, y_true, output, cols, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Single CTA per row, looping over column tiles with fp32 accumulation. We mask out `y_true == 0` to avoid `0 * log(0) = NaN`. Iter-0 picks `BLOCK_N/TILE=2048` to balance shared-mem usage with reuse across the cols range.
