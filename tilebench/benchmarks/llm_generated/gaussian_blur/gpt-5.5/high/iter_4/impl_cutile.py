import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _gaussian_blur_interior_static_vreuse_kernel(
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

    acc0 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc1 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc2 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc3 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc4 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc5 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc6 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc7 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)

    # Interior output is a shifted view of the input.  Use static ct.load from
    # per-column shifted slices instead of dynamic per-element gather; then reuse
    # each loaded input row across the eight output rows in this CTA.
    for t in range(0, BLOCK_H + KERNEL_ROWS - 1):
        in_r = row_block * BLOCK_H + t

        for kw in range(0, KERNEL_COLS):
            input_cols_view = input_2d.slice(1, kw, input_2d.shape[1] - (KERNEL_COLS - 1) + kw)
            vals = ct.astype(
                ct.load(
                    input_cols_view,
                    index=(in_r, col_block),
                    shape=(1, BLOCK_W),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False,
                    latency=1,
                ),
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
    ct.store(out_inner, index=(row_rel + 0, col_block), tile=ct.astype(acc0, input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 1, col_block), tile=ct.astype(acc1, input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 2, col_block), tile=ct.astype(acc2, input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 3, col_block), tile=ct.astype(acc3, input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 4, col_block), tile=ct.astype(acc4, input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 5, col_block), tile=ct.astype(acc5, input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 6, col_block), tile=ct.astype(acc6, input_2d.dtype))
    ct.store(out_inner, index=(row_rel + 7, col_block), tile=ct.astype(acc7, input_2d.dtype))


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
            _gaussian_blur_interior_static_vreuse_kernel,
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
            "interior_static_vreuse": 1,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
