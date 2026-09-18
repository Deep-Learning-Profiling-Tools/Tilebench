Looking at the trajectory: both backends are bandwidth-bound at 24-35% of roofline. There's no loop in the kernel (single load per row), so `num_stages` doesn't help. The key lever is **rows per CTA** to amortize launch overhead and improve memory coalescing. Iter 1 showed ROWS=8 beat ROWS=1, so let me push further to ROWS=16 for triton and add multi-row support to cutile.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _ce_kernel(logits_ptr, targets_ptr, out_ptr,
               batch_size, n_cols, stride_row,
               ROWS_PER_BLOCK: tl.constexpr,
               BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    row_mask = rows < batch_size

    cols = tl.arange(0, BLOCK_N)
    col_mask = cols < n_cols

    offs = rows[:, None] * stride_row + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    x = tl.load(logits_ptr + offs, mask=mask, other=-float('inf')).to(tl.float32)

    m = tl.max(x, axis=1)
    z = tl.exp(x - m[:, None])
    s = tl.sum(z, axis=1)
    log_s = tl.log(s)

    targets = tl.load(targets_ptr + rows, mask=row_mask, other=0)
    target_offs = rows * stride_row + targets
    target_logit = tl.load(logits_ptr + target_offs, mask=row_mask, other=0.0).to(tl.float32)

    loss = -(target_logit - m) + log_s
    tl.store(out_ptr + rows, loss.to(out_ptr.dtype.element_ty), mask=row_mask)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)

    BLOCK_N = triton.next_power_of_2(num_classes)
    if BLOCK_N < 256:
        BLOCK_N = 256

    ROWS_PER_BLOCK = 16
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(batch_size, ROWS_PER_BLOCK),)
    _ce_kernel[grid](
        logits, targets, output,
        batch_size, num_classes, logits.stride(0),
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"ROWS_PER_BLOCK": ROWS_PER_BLOCK,
                      "BLOCK_N": BLOCK_N,
                      "num_warps": num_warps,
                      "num_stages": num_stages})
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


@ct.kernel(occupancy=4)
def _ce_kernel(logits, targets, output,
               TILE_N: ConstInt, ROWS: ConstInt):
    bid = ct.bid(0)
    x = ct.load(logits, index=(bid, 0), shape=(ROWS, TILE_N),
                padding_mode=ct.PaddingMode.NEG_INF)
    x_f32 = ct.astype(x, np.float32)

    m = ct.max(x_f32, axis=1, keepdims=True)          # [ROWS, 1]
    z = ct.exp(x_f32 - m)
    s = ct.sum(z, axis=1, keepdims=False)             # [ROWS]
    log_s = ct.log(s)
    m_flat = ct.reshape(m, (ROWS,))

    target_tile = ct.load(targets, index=(bid,), shape=(ROWS,))
    target_i32 = ct.astype(target_tile, np.int32)
    row_idx = ct.arange(ROWS, dtype=np.int32) + bid * ROWS
    target_logit = ct.gather(logits, (row_idx, target_i32))
    target_f32 = ct.astype(target_logit, np.float32)

    loss = -(target_f32 - m_flat) + log_s
    out_tile = ct.astype(loss, logits.dtype)
    ct.store(output, index=(bid,), tile=out_tile)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)
    stream = torch.cuda.current_stream()

    TILE_N = 1
    while TILE_N < num_classes:
        TILE_N <<= 1
    if TILE_N < 256:
        TILE_N = 256

    # Multi-row tile: pick ROWS that divides batch_size where possible.
    ROWS = 8
    while ROWS > 1 and (batch_size % ROWS) != 0:
        ROWS //= 2
    occupancy = 4

    grid = (batch_size // ROWS, 1, 1)
    ct.launch(stream, grid, _ce_kernel,
              (logits, targets, output, TILE_N, ROWS))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "ROWS": ROWS, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: increase work per CTA. Triton goes from 8→16 rows/CTA (640 CTAs for batch=10240, ~4 per SM on B200). cuTile now processes 8 rows per CTA via 2D tile load `(ROWS, TILE_N)` with gather for per-row target lookup.
