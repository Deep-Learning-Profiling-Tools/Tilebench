import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _gaussian_blur_interior_2d_kernel(
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
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)

    offs_r = tl.arange(0, BLOCK_H)[:, None]
    offs_c = tl.arange(0, BLOCK_W)[None, :]

    rows = pid_r * BLOCK_H + PAD_H + offs_r
    cols = pid_c * BLOCK_W + PAD_W + offs_c

    out_mask = (rows < (input_rows - PAD_H)) & (cols < (input_cols - PAD_W))
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for kh in tl.static_range(0, KERNEL_ROWS):
        in_r = rows + kh - PAD_H
        for kw in tl.static_range(0, KERNEL_COLS):
            in_c = cols + kw - PAD_W
            vals = tl.load(
                input_ptr + in_r * input_cols + in_c,
                mask=out_mask,
                other=0.0,
                cache_modifier=".ca",
            ).to(tl.float32)
            kval = tl.load(kernel_ptr + kh * KERNEL_COLS + kw).to(tl.float32)
            acc += vals * kval

    tl.store(output_ptr + rows * input_cols + cols, acc, mask=out_mask)


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

    BLOCK_H = 16
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
        _gaussian_blur_interior_2d_kernel[grid_interior](
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
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
