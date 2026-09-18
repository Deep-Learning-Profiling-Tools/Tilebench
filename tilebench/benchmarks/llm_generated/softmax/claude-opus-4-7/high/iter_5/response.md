Looking at the trajectory, my triton is hovering near 56% with the same config; cutile is stuck around 18%. Let me try a different angle:

- **Triton**: try `grid = (n_rows,)` (one CTA per row, no persistent loop) — with only 2048 rows × 16 warps this should give the scheduler more flexibility.
- **cuTile**: try higher occupancy with `latency=1` hint to reduce DRAM staging.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                    n_rows, n_cols,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols,
                mask=mask, other=-float('inf')).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 512:
        num_warps = 4
    else:
        num_warps = 2

    num_stages = 2

    grid = (n_rows,)
    _softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "num_programs": n_rows,
    })
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

LOG2_E = 1.4426950408889634


@ct.kernel(occupancy=4)
def _softmax_kernel(x, output, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.NEG_INF,
                   latency=1)
    tile_f32 = ct.astype(tile, np.float32)
    max_val = ct.max(tile_f32, axis=1, keepdims=True)
    exp_tile = ct.exp2((tile_f32 - max_val) * LOG2_E, flush_to_zero=True)
    sum_val = ct.sum(exp_tile, axis=1, keepdims=True)
    out_tile = exp_tile / sum_val
    ct.store(output, index=(row, 0), tile=ct.astype(out_tile, x.dtype),
             latency=1)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    stream = torch.cuda.current_stream()

    TILE = 1 << (n_cols - 1).bit_length()
    if TILE < 256:
        TILE = 256

    grid = (n_rows, 1, 1)
    ct.launch(stream, grid, _softmax_kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 4})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: Switched Triton to a non-persistent grid of one CTA per row (simpler scheduling, more parallelism for 2048 rows on 148 SMs). For cuTile, bumped occupancy to 4 and added `latency=1` hints to favor low-latency DRAM staging since the kernel is purely bandwidth-bound.
