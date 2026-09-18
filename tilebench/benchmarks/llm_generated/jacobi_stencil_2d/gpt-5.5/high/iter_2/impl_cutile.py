import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _jacobi_interior_kernel(input, output,
                            ROWS: ConstInt, COLS: ConstInt,
                            BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    top_view = input.slice(0, 0, ROWS - 2).slice(1, 1, COLS - 1)
    bottom_view = input.slice(0, 2, ROWS).slice(1, 1, COLS - 1)
    left_view = input.slice(0, 1, ROWS - 1).slice(1, 0, COLS - 2)
    right_view = input.slice(0, 1, ROWS - 1).slice(1, 2, COLS)
    out_view = output.slice(0, 1, ROWS - 1).slice(1, 1, COLS - 1)

    acc = ct.astype(
        ct.load(
            top_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )
    acc = acc + ct.astype(
        ct.load(
            bottom_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )
    acc = acc + ct.astype(
        ct.load(
            left_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )
    acc = acc + ct.astype(
        ct.load(
            right_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )

    ct.store(
        out_view,
        index=(bid_m, bid_n),
        tile=ct.astype(acc * 0.25, input.dtype),
        latency=1,
        allow_tma=False,
    )


@ct.kernel
def _copy_rows_kernel(input, output,
                      ROWS: ConstInt, COLS: ConstInt,
                      BLOCK_N: ConstInt):
    bid_n = ct.bid(0)

    top = ct.load(
        input,
        index=(0, bid_n),
        shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        output,
        index=(0, bid_n),
        tile=top,
        latency=1,
        allow_tma=False,
    )

    bottom_in = input.slice(0, ROWS - 1, ROWS)
    bottom_out = output.slice(0, ROWS - 1, ROWS)
    bottom = ct.load(
        bottom_in,
        index=(0, bid_n),
        shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        bottom_out,
        index=(0, bid_n),
        tile=bottom,
        latency=1,
        allow_tma=False,
    )


@ct.kernel
def _copy_cols_kernel(input, output,
                      ROWS: ConstInt, COLS: ConstInt,
                      BLOCK_M: ConstInt):
    bid_m = ct.bid(0)

    left_in = input.slice(0, 1, ROWS - 1).slice(1, 0, 1)
    left_out = output.slice(0, 1, ROWS - 1).slice(1, 0, 1)
    left = ct.load(
        left_in,
        index=(bid_m, 0),
        shape=(BLOCK_M, 1),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        left_out,
        index=(bid_m, 0),
        tile=left,
        latency=1,
        allow_tma=False,
    )

    right_in = input.slice(0, 1, ROWS - 1).slice(1, COLS - 1, COLS)
    right_out = output.slice(0, 1, ROWS - 1).slice(1, COLS - 1, COLS)
    right = ct.load(
        right_in,
        index=(bid_m, 0),
        shape=(BLOCK_M, 1),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        right_out,
        index=(bid_m, 0),
        tile=right,
        latency=1,
        allow_tma=False,
    )


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_M = 8
    BLOCK_N = 1024
    BOUNDARY_N = 1024
    BOUNDARY_M = 256
    occupancy = 2
    boundary_occupancy = 4

    row_kernel = _copy_rows_kernel.with_hints(occupancy=boundary_occupancy)
    ct.launch(
        stream,
        (ct.cdiv(cols, BOUNDARY_N), 1, 1),
        row_kernel,
        (input, output, rows, cols, BOUNDARY_N),
    )

    if rows > 2:
        col_kernel = _copy_cols_kernel.with_hints(occupancy=boundary_occupancy)
        ct.launch(
            stream,
            (ct.cdiv(rows - 2, BOUNDARY_M), 1, 1),
            col_kernel,
            (input, output, rows, cols, BOUNDARY_M),
        )

    if rows > 2 and cols > 2:
        interior_kernel = _jacobi_interior_kernel.with_hints(occupancy=occupancy)
        grid = (ct.cdiv(rows - 2, BLOCK_M), ct.cdiv(cols - 2, BLOCK_N), 1)
        ct.launch(
            stream,
            grid,
            interior_kernel,
            (input, output, rows, cols, BLOCK_M, BLOCK_N),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BOUNDARY_N": BOUNDARY_N,
        "BOUNDARY_M": BOUNDARY_M,
        "occupancy": occupancy,
        "boundary_occupancy": boundary_occupancy,
        "sliced_interior": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
