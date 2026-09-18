import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _accum_rows4(a0, a1, a2, a3, x0, x1, x2, x3, x4, x5, w0, w1, w2):
    b0 = a0 + x0 * w0 + x1 * w1 + x2 * w2
    b1 = a1 + x1 * w0 + x2 * w1 + x3 * w2
    b2 = a2 + x2 * w0 + x3 * w1 + x4 * w2
    b3 = a3 + x3 * w0 + x4 * w1 + x5 * w2
    return b0, b1, b2, b3


@ct.kernel
def _conv3d_3x3x3_d4_rows4_kernel(input, kernel, output,
                                  ROWS_PER_BLOCK: ConstInt,
                                  DEPTHS_PER_BLOCK: ConstInt,
                                  TILE: ConstInt):
    bid_c = ct.bid(0)
    bid_rb = ct.bid(1)
    bid_db = ct.bid(2)

    row0 = bid_rb * ROWS_PER_BLOCK
    od0 = bid_db * DEPTHS_PER_BLOCK

    acc00 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc01 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc02 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc03 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    acc10 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc11 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc12 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc13 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    acc20 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc21 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc22 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc23 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    acc30 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc31 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc32 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc33 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    for p in range(0, 6):
        for kc in range(0, 3):
            shifted = input.slice(2, kc, input.shape[2])

            x0 = ct.astype(
                ct.load(
                    shifted, index=(od0 + p, row0 + 0, bid_c),
                    shape=(1, 1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False,
                    latency=1,
                ),
                np.float32,
            )
            x1 = ct.astype(
                ct.load(
                    shifted, index=(od0 + p, row0 + 1, bid_c),
                    shape=(1, 1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False,
                    latency=1,
                ),
                np.float32,
            )
            x2 = ct.astype(
                ct.load(
                    shifted, index=(od0 + p, row0 + 2, bid_c),
                    shape=(1, 1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False,
                    latency=1,
                ),
                np.float32,
            )
            x3 = ct.astype(
                ct.load(
                    shifted, index=(od0 + p, row0 + 3, bid_c),
                    shape=(1, 1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False,
                    latency=1,
                ),
                np.float32,
            )
            x4 = ct.astype(
                ct.load(
                    shifted, index=(od0 + p, row0 + 4, bid_c),
                    shape=(1, 1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False,
                    latency=1,
                ),
                np.float32,
            )
            x5 = ct.astype(
                ct.load(
                    shifted, index=(od0 + p, row0 + 5, bid_c),
                    shape=(1, 1, TILE),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False,
                    latency=1,
                ),
                np.float32,
            )

            if p < 3:
                w0 = ct.astype(
                    ct.load(
                        kernel, index=(p, 0, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w1 = ct.astype(
                    ct.load(
                        kernel, index=(p, 1, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w2 = ct.astype(
                    ct.load(
                        kernel, index=(p, 2, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                acc00, acc01, acc02, acc03 = _accum_rows4(
                    acc00, acc01, acc02, acc03, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

            if p >= 1 and p <= 3:
                kd = p - 1
                w0 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 0, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w1 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 1, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w2 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 2, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                acc10, acc11, acc12, acc13 = _accum_rows4(
                    acc10, acc11, acc12, acc13, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

            if p >= 2 and p <= 4:
                kd = p - 2
                w0 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 0, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w1 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 1, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w2 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 2, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                acc20, acc21, acc22, acc23 = _accum_rows4(
                    acc20, acc21, acc22, acc23, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

            if p >= 3:
                kd = p - 3
                w0 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 0, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w1 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 1, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w2 = ct.astype(
                    ct.load(
                        kernel, index=(kd, 2, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                acc30, acc31, acc32, acc33 = _accum_rows4(
                    acc30, acc31, acc32, acc33, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

    ct.store(output, index=(od0 + 0, row0 + 0, bid_c),
             tile=ct.astype(acc00, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 0, row0 + 1, bid_c),
             tile=ct.astype(acc01, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 0, row0 + 2, bid_c),
             tile=ct.astype(acc02, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 0, row0 + 3, bid_c),
             tile=ct.astype(acc03, output.dtype), allow_tma=False, latency=1)

    ct.store(output, index=(od0 + 1, row0 + 0, bid_c),
             tile=ct.astype(acc10, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 1, row0 + 1, bid_c),
             tile=ct.astype(acc11, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 1, row0 + 2, bid_c),
             tile=ct.astype(acc12, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 1, row0 + 3, bid_c),
             tile=ct.astype(acc13, output.dtype), allow_tma=False, latency=1)

    ct.store(output, index=(od0 + 2, row0 + 0, bid_c),
             tile=ct.astype(acc20, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 2, row0 + 1, bid_c),
             tile=ct.astype(acc21, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 2, row0 + 2, bid_c),
             tile=ct.astype(acc22, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 2, row0 + 3, bid_c),
             tile=ct.astype(acc23, output.dtype), allow_tma=False, latency=1)

    ct.store(output, index=(od0 + 3, row0 + 0, bid_c),
             tile=ct.astype(acc30, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 3, row0 + 1, bid_c),
             tile=ct.astype(acc31, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 3, row0 + 2, bid_c),
             tile=ct.astype(acc32, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 3, row0 + 3, bid_c),
             tile=ct.astype(acc33, output.dtype), allow_tma=False, latency=1)


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

    TILE = 256

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        ROWS_PER_BLOCK = 4
        DEPTHS_PER_BLOCK = 4
        occupancy = 2
        grid = (
            ct.cdiv(output_cols, TILE),
            ct.cdiv(output_rows, ROWS_PER_BLOCK),
            ct.cdiv(output_depth, DEPTHS_PER_BLOCK),
        )
        kernel_fn = _conv3d_3x3x3_d4_rows4_kernel.with_hints(occupancy=occupancy)
        ct.launch(
            stream,
            grid,
            kernel_fn,
            (input_3d, kernel_3d, output, ROWS_PER_BLOCK, DEPTHS_PER_BLOCK, TILE),
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": TILE,
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
            "DEPTHS_PER_BLOCK": DEPTHS_PER_BLOCK,
            "occupancy": occupancy,
            "variant": "3x3x3_d4_rows4_direct",
        })
    else:
        ROWS_PER_BLOCK = 1
        occupancy = 4
        grid = (ct.cdiv(output_cols, TILE), output_rows, output_depth)
        kernel_fn = _conv3d_generic_kernel.with_hints(occupancy=occupancy)
        ct.launch(
            stream,
            grid,
            kernel_fn,
            (input_3d, kernel_3d, output,
             kernel_depth, kernel_rows, kernel_cols, TILE),
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
