Looking at the trajectory:
- Iter 3 broke triton (exp2 caused compile failure). Revert to iter 0's safe `tl.exp` approach.
- Cutile best was iter 3 with occupancy=2 (17.5%). Try occupancy=3 to push more parallelism within shmem budget for fp32.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                    n_rows, n_cols,
                    BLOCK_SIZE: tl.constexpr,
                    NUM_STAGES: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    for row in tl.range(row_start, n_rows, row_step, num_stages=NUM_STAGES):
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

    NUM_STAGES = 2

    NUM_SM = 148
    occupancy = 2
    num_programs = min(NUM_SM * occupancy, n_rows)

    grid = (num_programs,)
    _softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_STAGES=NUM_STAGES,
        num_warps=num_warps,
        num_stages=NUM_STAGES,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": NUM_STAGES,
        "num_programs": num_programs,
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


@ct.kernel(occupancy=2)
def _softmax_kernel_fp32(x, output, n_cols: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.NEG_INF)
    tile_f32 = ct.astype(tile, np.float32)
    max_val = ct.max(tile_f32, axis=1, keepdims=True)
    exp_tile = ct.exp2((tile_f32 - max_val) * LOG2_E, flush_to_zero=True)
    sum_val = ct.sum(exp_tile, axis=1, keepdims=True)
    out_tile = exp_tile / sum_val
    ct.store(output, index=(row, 0), tile=ct.astype(out_tile, x.dtype))


@ct.kernel(occupancy=3)
def _softmax_kernel_fp16(x, output, n_cols: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.NEG_INF)
    tile_f32 = ct.astype(tile, np.float32)
    max_val = ct.max(tile_f32, axis=1, keepdims=True)
    exp_tile = ct.exp2((tile_f32 - max_val) * LOG2_E, flush_to_zero=True)
    sum_val = ct.sum(exp_tile, axis=1, keepdims=True)
    out_tile = exp_tile / sum_val
    ct.store(output, index=(row, 0), tile=ct.astype(out_tile, x.dtype))


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    stream = torch.cuda.current_stream()

    TILE = 1 << (n_cols - 1).bit_length()
    if TILE < 256:
        TILE = 256

    grid = (n_rows, 1, 1)

    if x.dtype == torch.float32:
        ct.launch(stream, grid, _softmax_kernel_fp32, (x, output, n_cols, TILE))
        occ = 2
    else:
        ct.launch(stream, grid, _softmax_kernel_fp16, (x, output, n_cols, TILE))
        occ = 3

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occ})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted triton to iter-0 baseline (verify-clean, 56.2%). For cutile, increased fp16 occupancy to 3 (smaller per-row shmem footprint allows more concurrent CTAs) while keeping fp32 at occupancy=2 to fit shmem budget.
