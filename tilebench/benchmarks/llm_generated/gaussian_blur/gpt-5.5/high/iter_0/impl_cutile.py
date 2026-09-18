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
