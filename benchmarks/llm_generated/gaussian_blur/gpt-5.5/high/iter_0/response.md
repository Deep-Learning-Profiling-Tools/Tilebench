```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _gaussian_blur_interior_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_rows,
    input_cols,
    KERNEL_ROWS: tl.constexpr,
    KERNEL_COLS: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)

    row = pid_r + PAD_H
    offs = tl.arange(0, BLOCK_W)
    cols = pid_c * BLOCK_W + PAD_W + offs
    col_mask = cols < (input_cols - PAD_W)

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for kh in tl.static_range(0, KERNEL_ROWS):
        in_r = row + kh - PAD_H
        for kw in tl.static_range(0, KERNEL_COLS):
            in_c = cols + kw - PAD_W
            vals = tl.load(
                input_ptr + in_r * input_cols + in_c,
                mask=col_mask,
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            kval = tl.load(kernel_ptr + kh * KERNEL_COLS + kw).to(tl.float32)
            acc += vals * kval

    tl.store(output_ptr + row * input_cols + cols, acc, mask=col_mask)


@triton.jit
def _gaussian_blur_boundary_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_rows,
    input_cols,
    n_boundary,
    KERNEL_ROWS: tl.constexpr,
    KERNEL_COLS: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid_out = offs < n_boundary

    top_count = PAD_H * input_cols
    bottom_start = top_count
    middle_start = top_count + top_count

    is_top = offs < top_count
    is_bottom = (offs >= bottom_start) & (offs < middle_start)

    top_r = offs // input_cols
    top_c = offs - top_r * input_cols

    bidx = offs - bottom_start
    bottom_r0 = bidx // input_cols
    bottom_r = input_rows - PAD_H + bottom_r0
    bottom_c = bidx - bottom_r0 * input_cols

    midx = offs - middle_start
    side_width = 2 * PAD_W
    mid_r0 = midx // side_width
    side_pos = midx - mid_r0 * side_width
    mid_r = PAD_H + mid_r0
    mid_c = tl.where(side_pos < PAD_W, side_pos, input_cols - PAD_W + (side_pos - PAD_W))

    row = tl.where(is_top, top_r, tl.where(is_bottom, bottom_r, mid_r))
    col = tl.where(is_top, top_c, tl.where(is_bottom, bottom_c, mid_c))

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kh in tl.static_range(0, KERNEL_ROWS):
        in_r = row + kh - PAD_H
        row_ok = (in_r >= 0) & (in_r < input_rows)
        for kw in tl.static_range(0, KERNEL_COLS):
            in_c = col + kw - PAD_W
            load_mask = valid_out & row_ok & (in_c >= 0) & (in_c < input_cols)
            vals = tl.load(
                input_ptr + in_r * input_cols + in_c,
                mask=load_mask,
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            kval = tl.load(kernel_ptr + kh * KERNEL_COLS + kw).to(tl.float32)
            acc += vals * kval

    tl.store(output_ptr + row * input_cols + col, acc, mask=valid_out)


def run(input, kernel, input_rows, input_cols, kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)

    H = int(input_rows)
    W = int(input_cols)
    KR = int(kernel_rows)
    KC = int(kernel_cols)
    PAD_H = KR // 2
    PAD_W = KC // 2

    BLOCK_W = 256
    BOUNDARY_BLOCK = 256
    num_warps = 4
    boundary_num_warps = 4
    num_stages = 3

    if H > 2 * PAD_H and W > 2 * PAD_W:
        grid_interior = (triton.cdiv(W - 2 * PAD_W, BLOCK_W), H - 2 * PAD_H)
        _gaussian_blur_interior_kernel[grid_interior](
            input,
            kernel,
            output,
            H,
            W,
            KERNEL_ROWS=KR,
            KERNEL_COLS=KC,
            PAD_H=PAD_H,
            PAD_W=PAD_W,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    n_boundary = H * W
    if H > 2 * PAD_H and W > 2 * PAD_W:
        n_boundary = H * W - (H - 2 * PAD_H) * (W - 2 * PAD_W)

    if n_boundary > 0:
        grid_boundary = (triton.cdiv(n_boundary, BOUNDARY_BLOCK),)
        _gaussian_blur_boundary_kernel[grid_boundary](
            input,
            kernel,
            output,
            H,
            W,
            n_boundary,
            KERNEL_ROWS=KR,
            KERNEL_COLS=KC,
            PAD_H=PAD_H,
            PAD_W=PAD_W,
            BLOCK_SIZE=BOUNDARY_BLOCK,
            num_warps=boundary_num_warps,
            num_stages=num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_W": BLOCK_W,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "num_warps": num_warps,
            "boundary_num_warps": boundary_num_warps,
            "num_stages": num_stages,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
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
def _gaussian_blur_kernel(
    input_2d,
    kernel_2d,
    output_2d,
    KERNEL_ROWS: ConstInt,
    KERNEL_COLS: ConstInt,
    TILE: ConstInt,
):
    row = ct.bid(1)
    col_block = ct.bid(0)

    offs = ct.arange(TILE, dtype=np.int32)
    cols = col_block * TILE + offs

    acc = ct.full((TILE,), 0.0, dtype=np.float32)

    for kh in range(0, KERNEL_ROWS):
        in_r = row + kh - (KERNEL_ROWS // 2)
        for kw in range(0, KERNEL_COLS):
            in_c = cols + kw - (KERNEL_COLS // 2)
            vals = ct.astype(
                ct.gather(input_2d, (in_r, in_c), padding_value=0.0),
                np.float32,
            )
            kval = ct.astype(
                ct.load(kernel_2d, index=(kh, kw), shape=()),
                np.float32,
            )
            acc = acc + vals * kval

    out_tile = ct.reshape(ct.astype(acc, input_2d.dtype), (1, TILE))
    ct.store(output_2d, index=(row, col_block), tile=out_tile)


def run(input, kernel, input_rows, input_cols, kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)

    H = int(input_rows)
    W = int(input_cols)
    KR = int(kernel_rows)
    KC = int(kernel_cols)

    input_2d = input.view(H, W)
    kernel_2d = kernel.view(KR, KC)
    output_2d = output.view(H, W)

    TILE = 256
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(W, TILE), H, 1)
    launch_kernel = _gaussian_blur_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, launch_kernel, (input_2d, kernel_2d, output_2d, KR, KC, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "occupancy": occupancy,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
