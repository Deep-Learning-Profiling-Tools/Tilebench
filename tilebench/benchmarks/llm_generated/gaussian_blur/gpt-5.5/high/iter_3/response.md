```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _gaussian_blur_interior_sep_vreuse_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_rows,
    input_cols,
    KERNEL_ROWS: tl.constexpr,
    KERNEL_COLS: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tl.static_assert(BLOCK_H == 8)

    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)

    offs = tl.arange(0, BLOCK_W)
    base_out_row = pid_r * BLOCK_H + PAD_H
    cols = pid_c * BLOCK_W + PAD_W + offs
    col_mask = cols < (input_cols - PAD_W)

    center = tl.load(kernel_ptr + PAD_H * KERNEL_COLS + PAD_W).to(tl.float32)
    inv_center = 1.0 / center

    acc0 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc5 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc6 = tl.zeros((BLOCK_W,), dtype=tl.float32)
    acc7 = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # Gaussian kernels are separable.  Compute the 7-wide horizontal blur for
    # each source row once, then vertically reuse it for the 8 output rows.
    for t in tl.static_range(0, BLOCK_H + KERNEL_ROWS - 1):
        in_r = base_out_row + t - PAD_H
        row_ok = (in_r >= 0) & (in_r < input_rows)

        horiz = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for kw in tl.static_range(0, KERNEL_COLS):
            in_c = cols + kw - PAD_W
            vals = tl.load(
                input_ptr + in_r * input_cols + in_c,
                mask=row_ok & col_mask,
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            hval = tl.load(kernel_ptr + PAD_H * KERNEL_COLS + kw).to(tl.float32)
            horiz += vals * hval

        if t < KERNEL_ROWS:
            vval = tl.load(kernel_ptr + t * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc0 += horiz * vval
        if (t >= 1) and ((t - 1) < KERNEL_ROWS):
            vval = tl.load(kernel_ptr + (t - 1) * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc1 += horiz * vval
        if (t >= 2) and ((t - 2) < KERNEL_ROWS):
            vval = tl.load(kernel_ptr + (t - 2) * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc2 += horiz * vval
        if (t >= 3) and ((t - 3) < KERNEL_ROWS):
            vval = tl.load(kernel_ptr + (t - 3) * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc3 += horiz * vval
        if (t >= 4) and ((t - 4) < KERNEL_ROWS):
            vval = tl.load(kernel_ptr + (t - 4) * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc4 += horiz * vval
        if (t >= 5) and ((t - 5) < KERNEL_ROWS):
            vval = tl.load(kernel_ptr + (t - 5) * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc5 += horiz * vval
        if (t >= 6) and ((t - 6) < KERNEL_ROWS):
            vval = tl.load(kernel_ptr + (t - 6) * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc6 += horiz * vval
        if (t >= 7) and ((t - 7) < KERNEL_ROWS):
            vval = tl.load(kernel_ptr + (t - 7) * KERNEL_COLS + PAD_W).to(tl.float32) * inv_center
            acc7 += horiz * vval

    row0 = base_out_row + 0
    row1 = base_out_row + 1
    row2 = base_out_row + 2
    row3 = base_out_row + 3
    row4 = base_out_row + 4
    row5 = base_out_row + 5
    row6 = base_out_row + 6
    row7 = base_out_row + 7

    tl.store(output_ptr + row0 * input_cols + cols, acc0, mask=(row0 < (input_rows - PAD_H)) & col_mask)
    tl.store(output_ptr + row1 * input_cols + cols, acc1, mask=(row1 < (input_rows - PAD_H)) & col_mask)
    tl.store(output_ptr + row2 * input_cols + cols, acc2, mask=(row2 < (input_rows - PAD_H)) & col_mask)
    tl.store(output_ptr + row3 * input_cols + cols, acc3, mask=(row3 < (input_rows - PAD_H)) & col_mask)
    tl.store(output_ptr + row4 * input_cols + cols, acc4, mask=(row4 < (input_rows - PAD_H)) & col_mask)
    tl.store(output_ptr + row5 * input_cols + cols, acc5, mask=(row5 < (input_rows - PAD_H)) & col_mask)
    tl.store(output_ptr + row6 * input_cols + cols, acc6, mask=(row6 < (input_rows - PAD_H)) & col_mask)
    tl.store(output_ptr + row7 * input_cols + cols, acc7, mask=(row7 < (input_rows - PAD_H)) & col_mask)


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

    BLOCK_H = 8
    BLOCK_W = 256
    BOUNDARY_BLOCK = 256
    num_warps = 8
    boundary_num_warps = 4
    num_stages = 3

    if H > 2 * PAD_H and W > 2 * PAD_W:
        grid_interior = (
            triton.cdiv(W - 2 * PAD_W, BLOCK_W),
            triton.cdiv(H - 2 * PAD_H, BLOCK_H),
        )
        _gaussian_blur_interior_sep_vreuse_kernel[grid_interior](
            input,
            kernel,
            output,
            H,
            W,
            KERNEL_ROWS=KR,
            KERNEL_COLS=KC,
            PAD_H=PAD_H,
            PAD_W=PAD_W,
            BLOCK_H=BLOCK_H,
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
            "BLOCK_H": BLOCK_H,
            "BLOCK_W": BLOCK_W,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "num_warps": num_warps,
            "boundary_num_warps": boundary_num_warps,
            "num_stages": num_stages,
            "separable_rank1": 1,
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


@ct.kernel(occupancy=4)
def _gaussian_blur_interior_vreuse_kernel(
    input_2d,
    kernel_2d,
    output_2d,
    KERNEL_ROWS: ConstInt,
    KERNEL_COLS: ConstInt,
    BLOCK_H: ConstInt,
    BLOCK_W: ConstInt,
):
    row_block = ct.bid(1)
    col_block = ct.bid(0)

    pad_h = KERNEL_ROWS // 2
    pad_w = KERNEL_COLS // 2

    out_inner = output_2d.slice(0, pad_h, output_2d.shape[0] - pad_h)
    out_inner = out_inner.slice(1, pad_w, output_2d.shape[1] - pad_w)

    offs_c = ct.arange(BLOCK_W, dtype=np.int32)
    base_c_rel = col_block * BLOCK_W

    acc0 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)
    acc1 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)
    acc2 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)
    acc3 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)
    acc4 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)
    acc5 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)
    acc6 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)
    acc7 = ct.full((BLOCK_W,), 0.0, dtype=np.float32)

    for t in range(0, BLOCK_H + KERNEL_ROWS - 1):
        in_r = row_block * BLOCK_H + t

        for kw in range(0, KERNEL_COLS):
            in_c = base_c_rel + offs_c + kw
            vals = ct.astype(
                ct.gather(input_2d, (in_r, in_c), padding_value=0.0),
                np.float32,
            )

            if t < KERNEL_ROWS:
                kval = ct.astype(ct.load(kernel_2d, index=(t, kw), shape=()), np.float32)
                acc0 = acc0 + vals * kval
            if (t >= 1) and ((t - 1) < KERNEL_ROWS):
                kval = ct.astype(ct.load(kernel_2d, index=(t - 1, kw), shape=()), np.float32)
                acc1 = acc1 + vals * kval
            if (t >= 2) and ((t - 2) < KERNEL_ROWS):
                kval = ct.astype(ct.load(kernel_2d, index=(t - 2, kw), shape=()), np.float32)
                acc2 = acc2 + vals * kval
            if (t >= 3) and ((t - 3) < KERNEL_ROWS):
                kval = ct.astype(ct.load(kernel_2d, index=(t - 3, kw), shape=()), np.float32)
                acc3 = acc3 + vals * kval
            if (t >= 4) and ((t - 4) < KERNEL_ROWS):
                kval = ct.astype(ct.load(kernel_2d, index=(t - 4, kw), shape=()), np.float32)
                acc4 = acc4 + vals * kval
            if (t >= 5) and ((t - 5) < KERNEL_ROWS):
                kval = ct.astype(ct.load(kernel_2d, index=(t - 5, kw), shape=()), np.float32)
                acc5 = acc5 + vals * kval
            if (t >= 6) and ((t - 6) < KERNEL_ROWS):
                kval = ct.astype(ct.load(kernel_2d, index=(t - 6, kw), shape=()), np.float32)
                acc6 = acc6 + vals * kval
            if (t >= 7) and ((t - 7) < KERNEL_ROWS):
                kval = ct.astype(ct.load(kernel_2d, index=(t - 7, kw), shape=()), np.float32)
                acc7 = acc7 + vals * kval

    row_rel = row_block * BLOCK_H
    ct.store(out_inner, index=(row_rel + 0, col_block), tile=ct.astype(acc0[None, :], input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 1, col_block), tile=ct.astype(acc1[None, :], input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 2, col_block), tile=ct.astype(acc2[None, :], input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 3, col_block), tile=ct.astype(acc3[None, :], input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 4, col_block), tile=ct.astype(acc4[None, :], input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 5, col_block), tile=ct.astype(acc5[None, :], input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 6, col_block), tile=ct.astype(acc6[None, :], input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 7, col_block), tile=ct.astype(acc7[None, :], input_2d.dtype))


@ct.kernel(occupancy=4)
def _gaussian_blur_boundary_kernel(
    input_flat,
    kernel_flat,
    output_flat,
    INPUT_ROWS: ConstInt,
    INPUT_COLS: ConstInt,
    N_BOUNDARY: ConstInt,
    KERNEL_ROWS: ConstInt,
    KERNEL_COLS: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    bid = ct.bid(0)
    offs = bid * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=np.int32)
    valid_out = offs < N_BOUNDARY

    pad_h = KERNEL_ROWS // 2
    pad_w = KERNEL_COLS // 2

    top_count = pad_h * INPUT_COLS
    bottom_start = top_count
    middle_start = top_count + top_count

    is_top = offs < top_count
    is_bottom = (offs >= bottom_start) & (offs < middle_start)

    top_r = offs // INPUT_COLS
    top_c = offs - top_r * INPUT_COLS

    bidx = offs - bottom_start
    bottom_r0 = bidx // INPUT_COLS
    bottom_r = INPUT_ROWS - pad_h + bottom_r0
    bottom_c = bidx - bottom_r0 * INPUT_COLS

    midx = offs - middle_start
    side_width = 2 * pad_w
    mid_r0 = midx // side_width
    side_pos = midx - mid_r0 * side_width
    mid_r = pad_h + mid_r0
    mid_c = ct.where(side_pos < pad_w, side_pos, INPUT_COLS - pad_w + (side_pos - pad_w))

    row = ct.where(is_top, top_r, ct.where(is_bottom, bottom_r, mid_r))
    col = ct.where(is_top, top_c, ct.where(is_bottom, bottom_c, mid_c))

    acc = ct.full((BLOCK_SIZE,), 0.0, dtype=np.float32)

    for kh in range(0, KERNEL_ROWS):
        in_r = row + kh - pad_h
        row_ok = (in_r >= 0) & (in_r < INPUT_ROWS)
        for kw in range(0, KERNEL_COLS):
            in_c = col + kw - pad_w
            valid = valid_out & row_ok & (in_c >= 0) & (in_c < INPUT_COLS)
            in_idx = ct.where(valid, in_r * INPUT_COLS + in_c, -1)
            vals = ct.astype(ct.gather(input_flat, in_idx, padding_value=0.0), np.float32)
            kval = ct.astype(ct.load(kernel_flat, index=(kh * KERNEL_COLS + kw,), shape=()), np.float32)
            acc = acc + vals * kval

    out_idx = ct.where(valid_out, row * INPUT_COLS + col, -1)
    ct.scatter(output_flat, out_idx, ct.astype(acc, input_flat.dtype))


def run(input, kernel, input_rows, input_cols, kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)

    H = int(input_rows)
    W = int(input_cols)
    KR = int(kernel_rows)
    KC = int(kernel_cols)
    PAD_H = KR // 2
    PAD_W = KC // 2

    input_2d = input.view(H, W)
    kernel_2d = kernel.view(KR, KC)
    output_2d = output.view(H, W)

    BLOCK_H = 8
    BLOCK_W = 256
    BOUNDARY_BLOCK = 256
    occupancy = 4
    boundary_occupancy = 4

    stream = torch.cuda.current_stream()

    if H > 2 * PAD_H and W > 2 * PAD_W:
        grid_interior = (ct.cdiv(W - 2 * PAD_W, BLOCK_W), ct.cdiv(H - 2 * PAD_H, BLOCK_H), 1)
        ct.launch(
            stream,
            grid_interior,
            _gaussian_blur_interior_vreuse_kernel,
            (input_2d, kernel_2d, output_2d, KR, KC, BLOCK_H, BLOCK_W),
        )

    n_boundary = H * W
    if H > 2 * PAD_H and W > 2 * PAD_W:
        n_boundary = H * W - (H - 2 * PAD_H) * (W - 2 * PAD_W)

    if n_boundary > 0:
        grid_boundary = (ct.cdiv(n_boundary, BOUNDARY_BLOCK), 1, 1)
        ct.launch(
            stream,
            grid_boundary,
            _gaussian_blur_boundary_kernel,
            (input, kernel, output, H, W, n_boundary, KR, KC, BOUNDARY_BLOCK),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_H": BLOCK_H,
            "BLOCK_W": BLOCK_W,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "occupancy": occupancy,
            "boundary_occupancy": boundary_occupancy,
            "interior_vreuse": 1,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
