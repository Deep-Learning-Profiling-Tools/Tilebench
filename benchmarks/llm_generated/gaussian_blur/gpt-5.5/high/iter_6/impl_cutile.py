import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _gaussian_blur_vreuse_2d_kernel(
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

    acc0 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc1 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc2 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc3 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc4 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc5 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc6 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)
    acc7 = ct.full((1, BLOCK_W), 0.0, dtype=np.float32)

    # Reuse each gathered input row across all eight output rows in this CTA.
    # This keeps the exact 2D kernel weights, unlike the failed separable path.
    for t in range(0, BLOCK_H + KERNEL_ROWS - 1):
        in_r = row_block * BLOCK_H + t - pad_h

        for kw in range(0, KERNEL_COLS):
            in_c = cols + kw - pad_w
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
    ct.launch(
        stream,
        grid,
        _gaussian_blur_vreuse_2d_kernel,
        (input_2d, kernel_2d, output_2d, KR, KC, BLOCK_H, BLOCK_W),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_H": BLOCK_H,
            "BLOCK_W": BLOCK_W,
            "occupancy": occupancy,
            "vertical_reuse": 1,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
