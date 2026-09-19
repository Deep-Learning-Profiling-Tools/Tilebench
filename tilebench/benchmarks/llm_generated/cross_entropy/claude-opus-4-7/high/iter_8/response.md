Looking at the feedback, both backends are bandwidth-bound. The current kernel loads logits twice — once for the full row to compute logsumexp, then a second small load via `target_offs` to read the target logit. Since the target logit is already in the loaded tile, I can extract it via a masked reduction instead of doing a second memory access. This should improve effective bandwidth, especially for fp16 where the kernel underperforms.

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
    x = tl.load(logits_ptr + offs, mask=mask, other=-float('inf'),
                eviction_policy="evict_first").to(tl.float32)

    # Load target indices
    targets = tl.load(targets_ptr + rows, mask=row_mask, other=0,
                      eviction_policy="evict_first")

    # Extract target logit from already-loaded tile via masked reduction
    # (avoids a second global memory access)
    target_mask = cols[None, :] == targets[:, None]
    target_logit = tl.sum(tl.where(target_mask, x, 0.0), axis=1)

    # Log-sum-exp
    m = tl.max(x, axis=1)
    z = tl.exp(x - m[:, None])
    s = tl.sum(z, axis=1)
    log_s = tl.log(s)

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
    num_stages = 3

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
                      "num_stages": num_stages,
                      "eviction": "evict_first",
                      "opt": "target_via_masked_reduce"})
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


@ct.kernel(occupancy=8)
def _ce_kernel(logits, targets, output,
               NUM_CLASSES: ConstInt, TILE_N: ConstInt):
    row = ct.bid(0)
    x = ct.load(logits, index=(row, 0), shape=(1, TILE_N),
                padding_mode=ct.PaddingMode.NEG_INF,
                latency=1)
    x_f32 = ct.astype(x, np.float32)

    m = ct.max(x_f32)
    z = ct.exp(x_f32 - m)
    s = ct.sum(z)
    log_s = ct.log(s)

    # Load target index
    target_tile = ct.load(targets, index=(row,), shape=(1,))
    target_i32 = ct.astype(target_tile, np.int32)

    # Extract target logit via masked reduction from already-loaded tile
    cols = ct.arange(TILE_N, dtype=np.int32)
    cols_2d = ct.reshape(cols, (1, TILE_N))
    target_2d = ct.reshape(target_i32, (1, 1))
    mask = cols_2d == target_2d
    target_logit = ct.sum(ct.where(mask, x_f32, 0.0))

    loss = -(target_logit - m) + log_s
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

    grid = (batch_size, 1, 1)
    ct.launch(stream, grid, _ce_kernel,
              (logits, targets, output, num_classes, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "occupancy": 8,
                      "latency": 1,
                      "opt": "target_via_masked_reduce"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
