import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _gaussian_blur_2d_kernel(
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

    offs_r = ct.arange(BLOCK_H, dtype=np.int32)[:, None]
    offs_c = ct.arange(BLOCK_W, dtype=np.int32)[None, :]

    rows = row_block * BLOCK_H + offs_r
    cols = col_block * BLOCK_W + offs_c

    pad_h = KERNEL_ROWS // 2
    pad_w = KERNEL_COLS // 2

    acc = ct.full((BLOCK_H, BLOCK_W), 0.0, dtype=np.float32)

    for kh in range(0, KERNEL_ROWS):
        in_r = rows + kh - pad_h
        for kw in range(0, KERNEL_COLS):
            in_c = cols + kw - pad_w
            vals = ct.astype(
                ct.gather(input_2d, (in_r, in_c), padding_value=0.0),
                np.float32,
            )
            kval = ct.astype(
                ct.load(kernel_2d, index=(kh, kw), shape=()),
                np.float32,
            )
            acc = acc + vals * kval

    ct.store(output_2d, index=(row_block, col_block), tile=ct.astype(acc, input_2d.dtype))


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
    ct.launch(stream, grid, _gaussian_blur_2d_kernel, (input_2d, kernel_2d, output_2d, KR, KC, BLOCK_H, BLOCK_W))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_H": BLOCK_H,
            "BLOCK_W": BLOCK_W,
            "occupancy": occupancy,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
