Approach: One CTA per row (batch sample). Each CTA loads the entire row of `num_classes=512` logits, computes max for stability, log-sum-exp, gathers the target logit via the integer target index, and writes the per-sample loss.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _ce_kernel(logits_ptr, targets_ptr, out_ptr,
               n_cols, stride_row,
               BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    row_ptr = logits_ptr + row * stride_row
    x = tl.load(row_ptr + cols, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    z = tl.exp(x - m)
    s = tl.sum(z, axis=0)
    log_s = tl.log(s)

    target = tl.load(targets_ptr + row)
    target_logit = tl.load(row_ptr + target).to(tl.float32)
    loss = -(target_logit - m) + log_s
    tl.store(out_ptr + row, loss.to(out_ptr.dtype.element_ty))


def run(logits: torch.Tensor, targets: torch.Tensor):
    batch_size, num_classes = logits.shape
    output = torch.empty(batch_size, dtype=logits.dtype, device=logits.device)

    BLOCK_N = triton.next_power_of_2(num_classes)
    if BLOCK_N < 256:
        BLOCK_N = 256
    num_warps = 4
    num_stages = 1

    grid = (batch_size,)
    _ce_kernel[grid](
        logits, targets, output,
        num_classes, logits.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N,
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

    m = ct.max(x_f32)             # scalar (0D)
    z = ct.exp(x_f32 - m)
    s = ct.sum(z)                 # scalar (0D)
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
    occupancy = 4

    grid = (batch_size, 1, 1)
    kernel = _ce_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel,
              (logits, targets, output, num_classes, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
