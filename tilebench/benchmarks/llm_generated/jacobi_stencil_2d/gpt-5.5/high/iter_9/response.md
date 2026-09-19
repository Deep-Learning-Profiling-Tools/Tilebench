```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_full_kernel(input_ptr, output_ptr,
                        ROWS: tl.constexpr, COLS: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    row_start = 1 + pid_m * BLOCK_M
    col_start = 1 + pid_n * BLOCK_N

    top_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start - 1, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    bottom_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start + 1, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    left_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start - 1),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    right_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start + 1),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    out_blk = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.load(top_blk).to(tl.float32)
    acc += tl.load(bottom_blk).to(tl.float32)
    acc += tl.load(left_blk).to(tl.float32)
    acc += tl.load(right_blk).to(tl.float32)

    tl.store(out_blk, acc * 0.25)


@triton.jit
def _jacobi_right_tail_kernel(input_ptr, output_ptr,
                              ROWS: tl.constexpr, COLS: tl.constexpr,
                              N_START: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    tl.assume(pid_m >= 0)

    row_start = 1 + pid_m * BLOCK_M
    col_start = N_START

    top_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start - 1, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    bottom_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start + 1, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    left_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start - 1),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    right_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start + 1),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    out_blk = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.load(top_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    acc += tl.load(bottom_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    acc += tl.load(left_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    acc += tl.load(right_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    tl.store(out_blk, acc * 0.25, boundary_check=(0, 1))


@triton.jit
def _jacobi_bottom_tail_kernel(input_ptr, output_ptr,
                               ROWS: tl.constexpr, COLS: tl.constexpr,
                               M_START: tl.constexpr,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    tl.assume(pid_n >= 0)

    row_start = M_START
    col_start = 1 + pid_n * BLOCK_N

    top_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start - 1, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    bottom_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start + 1, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    left_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start - 1),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    right_blk = tl.make_block_ptr(
        base=input_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start + 1),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    out_blk = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS, COLS),
        strides=(COLS, 1),
        offsets=(row_start, col_start),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.load(top_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    acc += tl.load(bottom_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    acc += tl.load(left_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    acc += tl.load(right_blk, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

    tl.store(out_blk, acc * 0.25, boundary_check=(0, 1))


@triton.jit
def _boundary_kernel(input_ptr, output_ptr,
                     ROWS: tl.constexpr, COLS: tl.constexpr,
                     TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    row0 = offs < COLS
    row_last = (offs >= COLS) & (offs < 2 * COLS)

    side = offs - 2 * COLS
    side_row = 1 + (side // 2)
    side_col = tl.where((side % 2) == 0, 0, COLS - 1)

    idx0 = offs
    idx1 = (ROWS - 1) * COLS + (offs - COLS)
    idx2 = side_row * COLS + side_col
    idx = tl.where(row0, idx0, tl.where(row_last, idx1, idx2))

    val = tl.load(input_ptr + idx, mask=mask, other=0.0)
    tl.store(output_ptr + idx, val, mask=mask)


@triton.jit
def _copy_all_kernel(input_ptr, output_ptr,
                     N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    val = tl.load(input_ptr + offs, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask)


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_M = 4
    BLOCK_N = 1024
    num_warps = 8
    num_stages = 3
    BOUNDARY_BLOCK = 1024
    boundary_num_warps = 4

    if rows <= 2 or cols <= 2:
        n_elements = rows * cols
        _copy_all_kernel[(triton.cdiv(n_elements, BOUNDARY_BLOCK),)](
            input, output,
            N=n_elements,
            BLOCK=BOUNDARY_BLOCK,
            num_warps=boundary_num_warps,
            num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "num_warps": num_warps,
            "boundary_num_warps": boundary_num_warps,
            "num_stages": num_stages,
            "full_tail": 1,
            "block_ptr": 1,
        })
        return output

    interior_m = rows - 2
    interior_n = cols - 2

    full_m = interior_m // BLOCK_M
    full_n = interior_n // BLOCK_N
    tail_m = interior_m - full_m * BLOCK_M
    tail_n = interior_n - full_n * BLOCK_N

    if full_m > 0 and full_n > 0:
        _jacobi_full_kernel[(full_m, full_n)](
            input, output,
            ROWS=rows,
            COLS=cols,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if tail_n > 0:
        n_start = 1 + full_n * BLOCK_N
        _jacobi_right_tail_kernel[(triton.cdiv(interior_m, BLOCK_M),)](
            input, output,
            ROWS=rows,
            COLS=cols,
            N_START=n_start,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if tail_m > 0 and full_n > 0:
        m_start = 1 + full_m * BLOCK_M
        _jacobi_bottom_tail_kernel[(full_n,)](
            input, output,
            ROWS=rows,
            COLS=cols,
            M_START=m_start,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    boundary_total = 2 * cols + 2 * (rows - 2)
    _boundary_kernel[(triton.cdiv(boundary_total, BOUNDARY_BLOCK),)](
        input, output,
        ROWS=rows,
        COLS=cols,
        TOTAL=boundary_total,
        BLOCK=BOUNDARY_BLOCK,
        num_warps=boundary_num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "num_warps": num_warps,
        "boundary_num_warps": boundary_num_warps,
        "num_stages": num_stages,
        "full_tail": 1,
        "block_ptr": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
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
```
