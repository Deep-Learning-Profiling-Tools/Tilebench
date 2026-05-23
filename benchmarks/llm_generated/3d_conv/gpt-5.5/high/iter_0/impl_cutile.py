import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _conv3d_3x3x3_kernel(input, kernel, output, TILE: ConstInt):
    bid_c = ct.bid(0)
    row0 = ct.bid(1) * 2
    od = ct.bid(2)

    acc0 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc1 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    for kd in range(0, 3):
        for kc in range(0, 3):
            shifted = input.slice(2, kc, input.shape[2])

            x0 = ct.astype(ct.load(
                shifted, index=(od + kd, row0 + 0, bid_c), shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
            ), np.float32)
            x1 = ct.astype(ct.load(
                shifted, index=(od + kd, row0 + 1, bid_c), shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
            ), np.float32)
            x2 = ct.astype(ct.load(
                shifted, index=(od + kd, row0 + 2, bid_c), shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
            ), np.float32)
            x3 = ct.astype(ct.load(
                shifted, index=(od + kd, row0 + 3, bid_c), shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
            ), np.float32)

            w0 = ct.astype(ct.load(
                kernel, index=(kd, 0, kc), shape=(1, 1, 1),
                padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
            ), np.float32)
            w1 = ct.astype(ct.load(
                kernel, index=(kd, 1, kc), shape=(1, 1, 1),
                padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
            ), np.float32)
            w2 = ct.astype(ct.load(
                kernel, index=(kd, 2, kc), shape=(1, 1, 1),
                padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
            ), np.float32)

            acc0 = acc0 + x0 * w0 + x1 * w1 + x2 * w2
            acc1 = acc1 + x1 * w0 + x2 * w1 + x3 * w2

    ct.store(output, index=(od, row0, bid_c),
             tile=ct.astype(acc0, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od, row0 + 1, bid_c),
             tile=ct.astype(acc1, output.dtype), allow_tma=False, latency=1)


@ct.kernel
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
                x = ct.astype(ct.load(
                    shifted, index=(od + kd, orow + kr, bid_c), shape=(1, 1, TILE),
                    padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
                ), np.float32)
                w = ct.astype(ct.load(
                    kernel, index=(kd, kr, kc), shape=(1, 1, 1),
                    padding_mode=ct.PaddingMode.ZERO, allow_tma=False, latency=1
                ), np.float32)
                acc = acc + x * w

    ct.store(output, index=(od, orow, bid_c),
             tile=ct.astype(acc, output.dtype), allow_tma=False, latency=1)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    input_3d = input.view(input_depth, input_rows, input_cols)
    kernel_3d = kernel.view(kernel_depth, kernel_rows, kernel_cols)
    output = torch.empty((output_depth, output_rows, output_cols),
                         device=input.device, dtype=input.dtype)

    TILE = 256
    occupancy = 4

    stream = torch.cuda.current_stream()

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        ROWS_PER_BLOCK = 2
        grid = (ct.cdiv(output_cols, TILE),
                ct.cdiv(output_rows, ROWS_PER_BLOCK),
                output_depth)
        kernel_fn = _conv3d_3x3x3_kernel.with_hints(occupancy=occupancy)
        ct.launch(stream, grid, kernel_fn, (input_3d, kernel_3d, output, TILE))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": TILE,
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
            "occupancy": occupancy,
            "variant": "3x3x3_rows2",
        })
    else:
        ROWS_PER_BLOCK = 1
        grid = (ct.cdiv(output_cols, TILE), output_rows, output_depth)
        kernel_fn = _conv3d_generic_kernel.with_hints(occupancy=occupancy)
        ct.launch(
            stream, grid, kernel_fn,
            (input_3d, kernel_3d, output,
             kernel_depth, kernel_rows, kernel_cols, TILE)
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": TILE,
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
            "KERNEL_DEPTH": kernel_depth,
            "KERNEL_ROWS": kernel_rows,
            "KERNEL_COLS": kernel_cols,
            "occupancy": occupancy,
            "variant": "generic",
        })

    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
