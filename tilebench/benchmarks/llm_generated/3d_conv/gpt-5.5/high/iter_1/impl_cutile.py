import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv3d_3x3x3_row_mma_kernel(input, kernel, output,
                                 BLOCK_R: ConstInt,
                                 BLOCK_COL: ConstInt,
                                 BLOCK_K: ConstInt):
    bid_c = ct.bid(0)
    bid_r = ct.bid(1)
    od = ct.bid(2)

    row_start = bid_r * BLOCK_R
    col_start = bid_c * BLOCK_COL

    rk = ct.arange(BLOCK_K, dtype=np.int32)[:, None]
    cc = ct.arange(BLOCK_COL, dtype=np.int32)[None, :]
    rr = ct.arange(BLOCK_R, dtype=np.int32)[:, None]
    kk = ct.arange(BLOCK_K, dtype=np.int32)[None, :]
    rel = kk - rr

    rows = row_start + rk
    cols = col_start + cc

    acc = ct.full((BLOCK_R, BLOCK_COL), 0.0, dtype=np.float32)

    for kd in range(0, 3):
        for kc in range(0, 3):
            x = ct.astype(
                ct.gather(
                    input,
                    (od + kd, rows, cols + kc),
                    padding_value=0,
                    check_bounds=True,
                    latency=1,
                ),
                np.float32,
            )

            w0 = ct.astype(
                ct.load(
                    kernel, index=(kd, 0, kc), shape=(1, 1, 1),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False, latency=1,
                ),
                np.float32,
            ).item()
            w1 = ct.astype(
                ct.load(
                    kernel, index=(kd, 1, kc), shape=(1, 1, 1),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False, latency=1,
                ),
                np.float32,
            ).item()
            w2 = ct.astype(
                ct.load(
                    kernel, index=(kd, 2, kc), shape=(1, 1, 1),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False, latency=1,
                ),
                np.float32,
            ).item()

            w = ct.full((BLOCK_R, BLOCK_K), 0.0, dtype=np.float32)
            w = ct.where(rel == 0, w0, w)
            w = ct.where(rel == 1, w1, w)
            w = ct.where(rel == 2, w2, w)

            acc = ct.mma(w, x, acc)

    ct.store(
        output,
        index=(od, bid_r, bid_c),
        tile=ct.astype(acc, output.dtype).reshape((1, BLOCK_R, BLOCK_COL)),
        allow_tma=False,
        latency=1,
    )


@ct.kernel(occupancy=4)
def _conv3d_generic_kernel(input, kernel, output,
                           KERNEL_DEPTH: ConstInt,
                           KERNEL_ROWS: ConstInt,
                           KERNEL_COLS: ConstInt,
                           TILE: ConstInt):
    bid_c = ct.bid(0)
    orow = ct.bid(1)
    od = ct.bid(2)

    acc = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    for kd in range(0, KERNEL_DEPTH):
        for kr in range(0, KERNEL_ROWS):
            for kc in range(0, KERNEL_COLS):
                shifted = input.slice(2, kc, input.shape[2])
                x = ct.astype(
                    ct.load(
                        shifted, index=(od + kd, orow + kr, bid_c),
                        shape=(1, 1, TILE),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w = ct.astype(
                    ct.load(
                        kernel, index=(kd, kr, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                acc = acc + x * w

    ct.store(
        output,
        index=(od, orow, bid_c),
        tile=ct.astype(acc, output.dtype),
        allow_tma=False,
        latency=1,
    )


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    input_3d = input.view(input_depth, input_rows, input_cols)
    kernel_3d = kernel.view(kernel_depth, kernel_rows, kernel_cols)
    output = torch.empty((output_depth, output_rows, output_cols),
                         device=input.device, dtype=input.dtype)

    stream = torch.cuda.current_stream()

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        BLOCK_R = 16
        BLOCK_COL = 128
        BLOCK_K = 32
        occupancy = 2

        grid = (
            ct.cdiv(output_cols, BLOCK_COL),
            ct.cdiv(output_rows, BLOCK_R),
            output_depth,
        )
        ct.launch(
            stream,
            grid,
            _conv3d_3x3x3_row_mma_kernel,
            (input_3d, kernel_3d, output, BLOCK_R, BLOCK_COL, BLOCK_K),
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_R": BLOCK_R,
            "BLOCK_COL": BLOCK_COL,
            "BLOCK_K": BLOCK_K,
            "occupancy": occupancy,
            "variant": "3x3x3_row_mma_sparse",
        })
    else:
        TILE = 256
        occupancy = 4

        grid = (ct.cdiv(output_cols, TILE), output_rows, output_depth)
        ct.launch(
            stream,
            grid,
            _conv3d_generic_kernel,
            (input_3d, kernel_3d, output,
             kernel_depth, kernel_rows, kernel_cols, TILE),
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": TILE,
            "ROWS_PER_BLOCK": 1,
            "KERNEL_DEPTH": kernel_depth,
            "KERNEL_ROWS": kernel_rows,
            "KERNEL_COLS": kernel_cols,
            "occupancy": occupancy,
            "variant": "generic",
        })

    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
