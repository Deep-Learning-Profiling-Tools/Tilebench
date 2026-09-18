import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _gaussian_blur_2d_patch_kernel(
    input_2d,
    kernel_2d,
    output_2d,
    KERNEL_ROWS: ConstInt,
    KERNEL_COLS: ConstInt,
    BLOCK_H: ConstInt,
    BLOCK_W: ConstInt,
    ROW_EXT: ConstInt,
    COL_EXT: ConstInt,
):
    row_block = ct.bid(1)
    col_block = ct.bid(0)

    pad_h = KERNEL_ROWS // 2
    pad_w = KERNEL_COLS // 2

    ext_r = ct.arange(ROW_EXT, dtype=np.int32)[:, None]
    ext_c = ct.arange(COL_EXT, dtype=np.int32)[None, :]

    base_r = row_block * BLOCK_H - pad_h
    base_c = col_block * BLOCK_W - pad_w

    patch = ct.astype(
        ct.gather(
            input_2d,
            (base_r + ext_r, base_c + ext_c),
            padding_value=0.0,
        ),
        np.float32,
    )

    acc = ct.full((BLOCK_H, BLOCK_W), 0.0, dtype=np.float32)

    for kh in range(0, KERNEL_ROWS):
        for kw in range(0, KERNEL_COLS):
            vals = ct.extract(patch, (kh, kw), shape=(BLOCK_H, BLOCK_W))
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
    ROW_EXT = 16
    COL_EXT = 512
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(W, BLOCK_W), ct.cdiv(H, BLOCK_H), 1)
    ct.launch(
        stream,
        grid,
        _gaussian_blur_2d_patch_kernel,
        (input_2d, kernel_2d, output_2d, KR, KC, BLOCK_H, BLOCK_W, ROW_EXT, COL_EXT),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_H": BLOCK_H,
            "BLOCK_W": BLOCK_W,
            "ROW_EXT": ROW_EXT,
            "COL_EXT": COL_EXT,
            "occupancy": occupancy,
            "patch_reuse": 1,
            "kernel_rows": KR,
            "kernel_cols": KC,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
