```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_full_stripe_kernel(input_ptr, output_ptr,
                               ROWS: tl.constexpr, COLS: tl.constexpr,
                               OUT_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_base = 1 + pid_m * OUT_M
    offs_n = 1 + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    prev = tl.load(input_ptr + (row_base - 1) * COLS + offs_n).to(tl.float32)
    curr = tl.load(input_ptr + row_base * COLS + offs_n).to(tl.float32)
    nxt = tl.load(input_ptr + (row_base + 1) * COLS + offs_n).to(tl.float32)

    for i in tl.static_range(0, OUT_M):
        r = row_base + i
        left = tl.load(input_ptr + r * COLS + offs_n - 1).to(tl.float32)
        right = tl.load(input_ptr + r * COLS + offs_n + 1).to(tl.float32)
        acc = prev + nxt
        acc += left
        acc += right
        tl.store(output_ptr + r * COLS + offs_n, acc * 0.25, cache_modifier=".cg")

        if i < OUT_M - 1:
            prev = curr
            curr = nxt
            nxt = tl.load(input_ptr + (row_base + i + 2) * COLS + offs_n).to(tl.float32)


@triton.jit
def _jacobi_right_tail_stripe_kernel(input_ptr, output_ptr,
                                     ROWS: tl.constexpr, COLS: tl.constexpr,
                                     N_START: tl.constexpr,
                                     OUT_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)

    row_base = 1 + pid_m * OUT_M
    offs_n = N_START + tl.arange(0, BLOCK_N)
    cmask = offs_n < COLS - 1

    prev_row = row_base - 1
    curr_row = row_base
    next_row = row_base + 1

    prev = tl.load(
        input_ptr + prev_row * COLS + offs_n,
        mask=(prev_row < ROWS) & cmask,
        other=0.0,
    ).to(tl.float32)
    curr = tl.load(
        input_ptr + curr_row * COLS + offs_n,
        mask=(curr_row < ROWS) & cmask,
        other=0.0,
    ).to(tl.float32)
    nxt = tl.load(
        input_ptr + next_row * COLS + offs_n,
        mask=(next_row < ROWS) & cmask,
        other=0.0,
    ).to(tl.float32)

    for i in tl.static_range(0, OUT_M):
        r = row_base + i
        row_ok = r < ROWS - 1
        mask = row_ok & cmask

        left = tl.load(input_ptr + r * COLS + offs_n - 1, mask=mask, other=0.0).to(tl.float32)
        right = tl.load(input_ptr + r * COLS + offs_n + 1, mask=mask, other=0.0).to(tl.float32)

        acc = prev + nxt
        acc += left
        acc += right
        tl.store(output_ptr + r * COLS + offs_n, acc * 0.25, mask=mask, cache_modifier=".cg")

        if i < OUT_M - 1:
            prev = curr
            curr = nxt
            nr = row_base + i + 2
            nxt = tl.load(
                input_ptr + nr * COLS + offs_n,
                mask=(nr < ROWS) & cmask,
                other=0.0,
            ).to(tl.float32)


@triton.jit
def _jacobi_bottom_tail_stripe_kernel(input_ptr, output_ptr,
                                      ROWS: tl.constexpr, COLS: tl.constexpr,
                                      M_START: tl.constexpr,
                                      OUT_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)

    row_base = M_START
    offs_n = 1 + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    prev_row = row_base - 1
    curr_row = row_base
    next_row = row_base + 1

    prev = tl.load(
        input_ptr + prev_row * COLS + offs_n,
        mask=prev_row < ROWS,
        other=0.0,
    ).to(tl.float32)
    curr = tl.load(
        input_ptr + curr_row * COLS + offs_n,
        mask=curr_row < ROWS,
        other=0.0,
    ).to(tl.float32)
    nxt = tl.load(
        input_ptr + next_row * COLS + offs_n,
        mask=next_row < ROWS,
        other=0.0,
    ).to(tl.float32)

    for i in tl.static_range(0, OUT_M):
        r = row_base + i
        row_ok = r < ROWS - 1

        left = tl.load(input_ptr + r * COLS + offs_n - 1, mask=row_ok, other=0.0).to(tl.float32)
        right = tl.load(input_ptr + r * COLS + offs_n + 1, mask=row_ok, other=0.0).to(tl.float32)

        acc = prev + nxt
        acc += left
        acc += right
        tl.store(output_ptr + r * COLS + offs_n, acc * 0.25, mask=row_ok, cache_modifier=".cg")

        if i < OUT_M - 1:
            prev = curr
            curr = nxt
            nr = row_base + i + 2
            nxt = tl.load(
                input_ptr + nr * COLS + offs_n,
                mask=nr < ROWS,
                other=0.0,
            ).to(tl.float32)


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
    tl.store(output_ptr + idx, val, mask=mask, cache_modifier=".cg")


@triton.jit
def _copy_all_kernel(input_ptr, output_ptr,
                     N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    val = tl.load(input_ptr + offs, mask=mask, other=0.0)
    tl.store(output_ptr + offs, val, mask=mask, cache_modifier=".cg")


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)

    OUT_M = 16
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
            "OUT_M": OUT_M,
            "BLOCK_N": BLOCK_N,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "num_warps": num_warps,
            "boundary_num_warps": boundary_num_warps,
            "num_stages": num_stages,
            "stripe_loop": 1,
            "store_cg": 1,
        })
        return output

    interior_m = rows - 2
    interior_n = cols - 2

    full_m = interior_m // OUT_M
    full_n = interior_n // BLOCK_N
    tail_m = interior_m - full_m * OUT_M
    tail_n = interior_n - full_n * BLOCK_N

    if full_m > 0 and full_n > 0:
        _jacobi_full_stripe_kernel[(full_m, full_n)](
            input, output,
            ROWS=rows,
            COLS=cols,
            OUT_M=OUT_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if tail_n > 0:
        n_start = 1 + full_n * BLOCK_N
        _jacobi_right_tail_stripe_kernel[(triton.cdiv(interior_m, OUT_M),)](
            input, output,
            ROWS=rows,
            COLS=cols,
            N_START=n_start,
            OUT_M=OUT_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if tail_m > 0 and full_n > 0:
        m_start = 1 + full_m * OUT_M
        _jacobi_bottom_tail_stripe_kernel[(full_n,)](
            input, output,
            ROWS=rows,
            COLS=cols,
            M_START=m_start,
            OUT_M=OUT_M,
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
        "OUT_M": OUT_M,
        "BLOCK_N": BLOCK_N,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "num_warps": num_warps,
        "boundary_num_warps": boundary_num_warps,
        "num_stages": num_stages,
        "stripe_loop": 1,
        "store_cg": 1,
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
def _jacobi_full_stripe_kernel(input, output,
                               ROWS: ConstInt, COLS: ConstInt,
                               N_FULL_END: ConstInt,
                               OUT_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    offs = ct.arange(BLOCK_N, dtype=np.int32)[None, :]
    c = 1 + bid_n * BLOCK_N + offs
    row_base = 1 + bid_m * OUT_M

    out_arr = output.slice(0, 1, ROWS - 1).slice(1, 1, N_FULL_END)

    prev = ct.astype(
        ct.gather(input, (row_base - 1, c), check_bounds=False, latency=1),
        np.float32,
    )
    curr = ct.astype(
        ct.gather(input, (row_base, c), check_bounds=False, latency=1),
        np.float32,
    )
    nxt = ct.astype(
        ct.gather(input, (row_base + 1, c), check_bounds=False, latency=1),
        np.float32,
    )

    for i in range(0, OUT_M):
        r = row_base + i

        left = ct.astype(
            ct.gather(input, (r, c - 1), check_bounds=False, latency=1),
            np.float32,
        )
        right = ct.astype(
            ct.gather(input, (r, c + 1), check_bounds=False, latency=1),
            np.float32,
        )

        avg = (prev + nxt + left + right) * 0.25
        ct.store(
            out_arr,
            index=(bid_m * OUT_M + i, bid_n),
            tile=ct.astype(avg, input.dtype),
            latency=1,
            allow_tma=False,
        )

        if i < OUT_M - 1:
            prev = curr
            curr = nxt
            nxt = ct.astype(
                ct.gather(input, (row_base + i + 2, c), check_bounds=False, latency=1),
                np.float32,
            )


@ct.kernel(occupancy=4)
def _jacobi_right_tail_stripe_kernel(input, output,
                                     ROWS: ConstInt, COLS: ConstInt,
                                     N_START: ConstInt,
                                     OUT_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)

    offs = ct.arange(BLOCK_N, dtype=np.int32)[None, :]
    c = N_START + offs
    row_base = 1 + bid_m * OUT_M

    prev = ct.astype(
        ct.gather(input, (row_base - 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    curr = ct.astype(
        ct.gather(input, (row_base, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    nxt = ct.astype(
        ct.gather(input, (row_base + 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )

    for i in range(0, OUT_M):
        r = row_base + i

        left = ct.astype(
            ct.gather(input, (r, c - 1), padding_value=0.0, check_bounds=True, latency=1),
            np.float32,
        )
        right = ct.astype(
            ct.gather(input, (r, c + 1), padding_value=0.0, check_bounds=True, latency=1),
            np.float32,
        )

        avg = (prev + nxt + left + right) * 0.25

        row_tile = c * 0 + r
        valid = (row_tile < ROWS - 1) & (c < COLS - 1)
        row_out = ct.where(valid, row_tile, ROWS)
        col_out = ct.where(valid, c, COLS)

        ct.scatter(
            output,
            (row_out, col_out),
            ct.astype(avg, input.dtype),
            check_bounds=True,
            latency=1,
        )

        if i < OUT_M - 1:
            prev = curr
            curr = nxt
            nxt = ct.astype(
                ct.gather(
                    input,
                    (row_base + i + 2, c),
                    padding_value=0.0,
                    check_bounds=True,
                    latency=1,
                ),
                np.float32,
            )


@ct.kernel(occupancy=4)
def _jacobi_bottom_tail_stripe_kernel(input, output,
                                      ROWS: ConstInt, COLS: ConstInt,
                                      M_START: ConstInt, N_FULL_END: ConstInt,
                                      OUT_M: ConstInt, BLOCK_N: ConstInt):
    bid_n = ct.bid(0)

    offs = ct.arange(BLOCK_N, dtype=np.int32)[None, :]
    c = 1 + bid_n * BLOCK_N + offs
    row_base = M_START

    out_arr = output.slice(0, 1, ROWS - 1).slice(1, 1, N_FULL_END)

    prev = ct.astype(
        ct.gather(input, (row_base - 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    curr = ct.astype(
        ct.gather(input, (row_base, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    nxt = ct.astype(
        ct.gather(input, (row_base + 1, c), padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )

    for i in range(0, OUT_M):
        r = row_base + i

        left = ct.astype(
            ct.gather(input, (r, c - 1), padding_value=0.0, check_bounds=True, latency=1),
            np.float32,
        )
        right = ct.astype(
            ct.gather(input, (r, c + 1), padding_value=0.0, check_bounds=True, latency=1),
            np.float32,
        )

        avg = (prev + nxt + left + right) * 0.25
        ct.store(
            out_arr,
            index=(M_START - 1 + i, bid_n),
            tile=ct.astype(avg, input.dtype),
            latency=1,
            allow_tma=False,
        )

        if i < OUT_M - 1:
            prev = curr
            curr = nxt
            nxt = ct.astype(
                ct.gather(
                    input,
                    (row_base + i + 2, c),
                    padding_value=0.0,
                    check_bounds=True,
                    latency=1,
                ),
                np.float32,
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

    OUT_M = 8
    BLOCK_N = 1024
    BOUNDARY_BLOCK = 1024
    occupancy = 4
    boundary_occupancy = 4

    if rows <= 2 or cols <= 2:
        grid = (ct.cdiv(rows, OUT_M), ct.cdiv(cols, BLOCK_N), 1)
        ct.launch(stream, grid, _copy_all_kernel, (input, output, rows, cols, OUT_M, BLOCK_N))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "OUT_M": OUT_M,
            "BLOCK_N": BLOCK_N,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "occupancy": occupancy,
            "boundary_occupancy": boundary_occupancy,
            "stripe_loop": 1,
        })
        return output

    interior_m = rows - 2
    interior_n = cols - 2

    full_m = interior_m // OUT_M
    full_n = interior_n // BLOCK_N
    tail_m = interior_m - full_m * OUT_M
    tail_n = interior_n - full_n * BLOCK_N

    if full_m > 0 and full_n > 0:
        n_full_end = 1 + full_n * BLOCK_N
        ct.launch(
            stream,
            (full_m, full_n, 1),
            _jacobi_full_stripe_kernel,
            (input, output, rows, cols, n_full_end, OUT_M, BLOCK_N),
        )

    if tail_n > 0:
        n_start = 1 + full_n * BLOCK_N
        ct.launch(
            stream,
            (ct.cdiv(interior_m, OUT_M), 1, 1),
            _jacobi_right_tail_stripe_kernel,
            (input, output, rows, cols, n_start, OUT_M, BLOCK_N),
        )

    if tail_m > 0 and full_n > 0:
        m_start = 1 + full_m * OUT_M
        n_full_end = 1 + full_n * BLOCK_N
        ct.launch(
            stream,
            (full_n, 1, 1),
            _jacobi_bottom_tail_stripe_kernel,
            (input, output, rows, cols, m_start, n_full_end, OUT_M, BLOCK_N),
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
        "OUT_M": OUT_M,
        "BLOCK_N": BLOCK_N,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "occupancy": occupancy,
        "boundary_occupancy": boundary_occupancy,
        "stripe_loop": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
