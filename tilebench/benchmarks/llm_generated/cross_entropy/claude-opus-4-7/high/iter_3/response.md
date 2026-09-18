Looking at the data: iter 2 regressed both backends. I'll revert toward iter 1 baselines. The problem is bandwidth-bound (small per-row 512-element tile), and fp16 is doing worse than fp32 relatively, suggesting overhead per row matters. Let me try smaller `ROWS_PER_BLOCK` with fewer warps for triton (more CTAs to hide latency, less reduction work per CTA), and revert cutile to the single-row config with higher occupancy.

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

    m = tl.max(x, axis=1)                       # [ROWS_PER_BLOCK]
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

    ROWS_PER_BLOCK = 4
    num_warps = 4
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


@ct.kernel
def _ce_kernel(logits, targets, output,
               NUM_CLASSES: ConstInt, TILE_N: ConstInt):
    row = ct.bid(0)
    x = ct.load(logits, index=(row, 0), shape=(1, TILE_N),
                padding_mode=ct.PaddingMode.NEG_INF)
    x_f32 = ct.astype(x, np.float32)

    m = ct.max(x_f32)
    z = ct.exp(x_f32 - m)
    s = ct.sum(z)
    log_s = ct.log(s)

    target_tile = ct.load(targets, index=(row,), shape=(1,))
    target_i32 = ct.astype(target_tile, np.int32)
    target_logit = ct.gather(logits, (row, target_i32))
    target_f32 = ct.astype(target_logit, np.float32)

    loss = -(target_f32 - m) + log_s
    out_tile = ct.astype(loss, logits.dtype)
    ct.store(output, index=(row,), tile=out_tile)


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)
    stream = torch.cuda.current_stream()

    TILE_N = 1
    while TILE_N < num_classes:
        TILE_N <<= 1
    if TILE_N < 256:
        TILE_N = 256

    occupancy = 8
    kernel = _ce_kernel.with_hints(occupancy=occupancy)

    grid = (batch_size, 1, 1)
    ct.launch(stream, grid, kernel,
              (logits, targets, output, num_classes, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: Reverted to single-row per CTA for cuTile with higher occupancy=8 (from 4) to maximize SM utilization on this bandwidth-bound 512-element-row workload. For Triton, reduced ROWS_PER_BLOCK from 8→4 and num_warps from 8→4 — more CTAs (2560 vs 1280) gives the scheduler more latency-hiding flexibility and reduces per-CTA work granularity.
