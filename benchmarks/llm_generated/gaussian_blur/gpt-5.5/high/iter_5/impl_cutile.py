import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _gaussian_blur_sep_vreuse_kernel(
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

    offs_c = ct.arange(BLOCK_W, dtype=np.int32)[None, :]
    cols = col_block * BLOCK_W + offs_c

    center = ct.astype(ct.load(kernel_2d, index=(pad_h, pad_w), shape=()), np.float32)

    acc0 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc1 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc2 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc3 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc4 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc5 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc6 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc7 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)

    # Single pass over the full image, including boundaries.  For Gaussian
    # kernels use separability and reuse each horizontal-filtered input row
    # across all eight output rows in this tile.
    for t in range(0, BLOCK_H + KERNEL_ROWS - 1):
        in_r = row_block * BLOCK_H + t - pad_h

        h = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
        for kw in range(0, KERNEL_COLS):
            in_c = cols + kw - pad_w
            vals = ct.astype(
                ct.gather(input_2d, (in_r, in_c), padding_value=0.0),
                np.float32,
            )
            rw = ct.astype(ct.load(kernel_2d, index=(pad_h, kw), shape=()), np.float32)
            h = h + vals * rw

        if t < KERNEL_ROWS:
            cw = ct.astype(ct.load(kernel_2d, index=(t, pad_w), shape=()), np.float32) / center
            acc0 = acc0 + h * cw
        if (t >= 1) and ((t - 1) < KERNEL_ROWS):
            cw = ct.astype(ct.load(kernel_2d, index=(t - 1, pad_w), shape=()), np.float32) / center
            acc1 = acc1 + h * cw
        if (t >= 2) and ((t - 2) < KERNEL_ROWS):
            cw = ct.astype(ct.load(kernel_2d, index=(t - 2, pad_w), shape=()), np.float32) / center
            acc2 = acc2 + h * cw
        if (t >= 3) and ((t - 3) < KERNEL_ROWS):
            cw = ct.astype(ct.load(kernel_2d, index=(t - 3, pad_w), shape=()), np.float32) / center
            acc3 = acc3 + h * cw
        if (t >= 4) and ((t - 4) < KERNEL_ROWS):
            cw = ct.astype(ct.load(kernel_2d, index=(t - 4, pad_w), shape=()), np.float32) / center
            acc4 = acc4 + h * cw
        if (t >= 5) and ((t - 5) < KERNEL_ROWS):
            cw = ct.astype(ct.load(kernel_2d, index=(t - 5, pad_w), shape=()), np.float32) / center
            acc5 = acc5 + h * cw
        if (t >= 6) and ((t - 6) < KERNEL_ROWS):
            cw = ct.astype(ct.load(kernel_2d, index=(t - 6, pad_w), shape=()), np.float32) / center
            acc6 = acc6 + h * cw
        if (t >= 7) and ((t - 7) < KERNEL_ROWS):
            cw = ct.astype(ct.load(kernel_2d, index=(t - 7, pad_w), shape=()), np.float32) / center
            acc7 = acc7 + h * cw

    row_base = row_block * BLOCK_H
    ct.store(output_2d, index=(row_base + 0, col_block), tile=ct.astype(acc0, input_2d.dtype))
    ct.store(output_2d, index=(row_base + 1, col_block), tile=ct.astype(acc1, input_2d.dtype))
    ct.store(output_2d, index=(row_base + 2, col_block), tile=ct.astype(acc2, input_2d.dtype))
    ct.store(output_2d, index=(row_base + 3, col_block), tile=ct.astype(acc3, input_2d.dtype))
    ct.store(output_2d, index=(row_base + 4, col_block), tile=ct.astype(acc4, input_2d.dtype))
    ct.store(output_2d, index=(row_base + 5, col_block), tile=ct.astype(acc5, input_2d.dtype))
    ct.store(output_2d, index=(row_base + 6, col_block), tile=ct.astype(acc6, input_2d.dtype))
    ct.store(output_2d, index=(row_base + 7, col_block), tile=ct.astype(acc7, input_2d.dtype))


def run(input, kernel, input_rows, input_cols, kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)

    H = int(input_rows)
    W = int(input_cols)
    KR = int(kernel_rows)
    KC = int(kernel_cols)

    input_2d = input.view(H, W)
    kernel_2d = kernel.view(KR, KC)
    output_2d = output.view(H, W)

    BLOCK_H = 8
    BLOCK_W = 256
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(W, BLOCK_W), ct.cdiv(H, BLOCK_H), 1)
    kernel_fn = _gaussian_blur_sep_vreuse_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel_fn,
        (input_2d, kernel_2d, output_2d, KR, KC, BLOCK_H, BLOCK_W),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_H": BLOCK_H,
            "BLOCK_W": BLOCK_W,
            "occupancy": occupancy,
            "separable_vreuse": 1,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
