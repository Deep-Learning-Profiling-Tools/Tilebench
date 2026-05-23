import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _jacobi_full_kernel(input, output,
                        ROWS: ConstInt, COLS: ConstInt,
                        BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)[:, None]
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)[None, :]

    r = 1 + bid_m * BLOCK_M + offs_m
    c = 1 + bid_n * BLOCK_N + offs_n

    top = ct.astype(
        ct.gather(input, (r - 1, c), check_bounds=False, latency=1),
        np.float32,
    )
    bottom = ct.astype(
        ct.gather(input, (r + 1, c), check_bounds=False, latency=1),
        np.float32,
    )
    left = ct.astype(
        ct.gather(input, (r, c - 1), check_bounds=False, latency=1),
        np.float32,
    )
    right = ct.astype(
        ct.gather(input, (r, c + 1), check_bounds=False, latency=1),
        np.float32,
    )

    avg = (top + bottom + left + right) * 0.25
    ct.scatter(
        output,
        (r, c),
        ct.astype(avg, input.dtype),
        check_bounds=False,
        latency=1,
    )


@ct.kernel(occupancy=4)
def _jacobi_right_tail_kernel(input, output,
                              ROWS: ConstInt, COLS: ConstInt,
                              N_START: ConstInt,
                              BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)[:, None]
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)[None, :]

    r = 1 + bid_m * BLOCK_M + offs_m
    c = N_START + offs_n

    top = ct.astype(
        ct.gather(input, (r - 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    bottom = ct.astype(
        ct.gather(input, (r + 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    left = ct.astype(
        ct.gather(input, (r, c - 1), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    right = ct.astype(
        ct.gather(input, (r, c + 1), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )

    avg = (top + bottom + left + right) * 0.25
    valid = (r < ROWS - 1) & (c < COLS - 1)

    row_out = ct.where(valid, r, ROWS)
    col_out = ct.where(valid, c, COLS)

    ct.scatter(
        output,
        (row_out, col_out),
        ct.astype(avg, input.dtype),
        check_bounds=True,
        latency=1,
    )


@ct.kernel(occupancy=4)
def _jacobi_bottom_tail_kernel(input, output,
                               ROWS: ConstInt, COLS: ConstInt,
                               M_START: ConstInt,
                               BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_n = ct.bid(0)

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)[:, None]
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)[None, :]

    r = M_START + offs_m
    c = 1 + bid_n * BLOCK_N + offs_n

    top = ct.astype(
        ct.gather(input, (r - 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    bottom = ct.astype(
        ct.gather(input, (r + 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    left = ct.astype(
        ct.gather(input, (r, c - 1), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    right = ct.astype(
        ct.gather(input, (r, c + 1), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )

    avg = (top + bottom + left + right) * 0.25
    valid = r < ROWS - 1

    row_out = ct.where(valid, r, ROWS)
    col_out = ct.where(valid, c, COLS)

    ct.scatter(
        output,
        (row_out, col_out),
        ct.astype(avg, input.dtype),
        check_bounds=True,
        latency=1,
    )


@ct.kernel(occupancy=4)
def _boundary_kernel(input, output,
                     ROWS: ConstInt, COLS: ConstInt,
                     TOTAL: ConstInt, BLOCK: ConstInt):
    bid = ct.bid(0)
    offs = bid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    valid = offs < TOTAL

    row0 = offs < COLS
    row_last = (offs >= COLS) & (offs < 2 * COLS)

    side = offs - 2 * COLS
    side_row = 1 + (side // 2)
    side_col = ct.where((side % 2) == 0, 0, COLS - 1)

    row = ct.where(row0, 0, ct.where(row_last, ROWS - 1, side_row))
    col = ct.where(row0, offs, ct.where(row_last, offs - COLS, side_col))

    row = ct.where(valid, row, ROWS)
    col = ct.where(valid, col, COLS)

    val = ct.gather(input, (row, col), padding_value=0.0, check_bounds=True, latency=1)
    ct.scatter(output, (row, col), val, check_bounds=True, latency=1)


@ct.kernel(occupancy=4)
def _copy_all_kernel(input, output,
                     ROWS: ConstInt, COLS: ConstInt,
                     BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    tile = ct.load(
        input,
        index=(bid_m, bid_n),
        shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        output,
        index=(bid_m, bid_n),
        tile=tile,
        latency=1,
        allow_tma=False,
    )


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_M = 4
    BLOCK_N = 1024
    BOUNDARY_BLOCK = 1024
    occupancy = 4
    boundary_occupancy = 4

    if rows <= 2 or cols <= 2:
        grid = (ct.cdiv(rows, BLOCK_M), ct.cdiv(cols, BLOCK_N), 1)
        ct.launch(stream, grid, _copy_all_kernel, (input, output, rows, cols, BLOCK_M, BLOCK_N))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "occupancy": occupancy,
            "boundary_occupancy": boundary_occupancy,
            "split_interior": 1,
            "full_scatter": 1,
        })
        return output

    interior_m = rows - 2
    interior_n = cols - 2

    full_m = interior_m // BLOCK_M
    full_n = interior_n // BLOCK_N
    tail_m = interior_m - full_m * BLOCK_M
    tail_n = interior_n - full_n * BLOCK_N

    if full_m > 0 and full_n > 0:
        ct.launch(
            stream,
            (full_m, full_n, 1),
            _jacobi_full_kernel,
            (input, output, rows, cols, BLOCK_M, BLOCK_N),
        )

    if tail_n > 0:
        n_start = 1 + full_n * BLOCK_N
        ct.launch(
            stream,
            (ct.cdiv(interior_m, BLOCK_M), 1, 1),
            _jacobi_right_tail_kernel,
            (input, output, rows, cols, n_start, BLOCK_M, BLOCK_N),
        )

    if tail_m > 0 and full_n > 0:
        m_start = 1 + full_m * BLOCK_M
        ct.launch(
            stream,
            (full_n, 1, 1),
            _jacobi_bottom_tail_kernel,
            (input, output, rows, cols, m_start, BLOCK_M, BLOCK_N),
        )

    boundary_total = 2 * cols + 2 * (rows - 2)
    ct.launch(
        stream,
        (ct.cdiv(boundary_total, BOUNDARY_BLOCK), 1, 1),
        _boundary_kernel,
        (input, output, rows, cols, boundary_total, BOUNDARY_BLOCK),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "occupancy": occupancy,
        "boundary_occupancy": boundary_occupancy,
        "split_interior": 1,
        "full_scatter": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
