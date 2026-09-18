```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_full_kernel(input_ptr, output_ptr,
                        ROWS: tl.constexpr, COLS: tl.constexpr,
                        N_BASE: tl.constexpr,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = 1 + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = N_BASE + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = offs_m[:, None]
    c = offs_n[None, :]
    base = r * COLS + c

    acc = tl.load(input_ptr + base - COLS, eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base + COLS, eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base - 1, eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base + 1, eviction_policy="evict_last").to(tl.float32)

    tl.store(output_ptr + base, acc * 0.25, cache_modifier=".cg")


@triton.jit
def _jacobi_n_range_kernel(input_ptr, output_ptr,
                           ROWS: tl.constexpr, COLS: tl.constexpr,
                           N_START: tl.constexpr, N_END: tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)

    offs_m = 1 + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = N_START + tl.arange(0, BLOCK_N)

    r = offs_m[:, None]
    c = offs_n[None, :]
    base = r * COLS + c
    mask = (r < ROWS - 1) & (c < N_END)

    acc = tl.load(input_ptr + base - COLS, mask=mask, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base + COLS, mask=mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base - 1, mask=mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base + 1, mask=mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)

    tl.store(output_ptr + base, acc * 0.25, mask=mask, cache_modifier=".cg")


@triton.jit
def _jacobi_bottom_tail_kernel(input_ptr, output_ptr,
                               ROWS: tl.constexpr, COLS: tl.constexpr,
                               M_START: tl.constexpr, N_BASE: tl.constexpr,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)

    offs_m = M_START + tl.arange(0, BLOCK_M)
    offs_n = N_BASE + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = offs_m[:, None]
    c = offs_n[None, :]
    base = r * COLS + c
    mask = r < ROWS - 1

    acc = tl.load(input_ptr + base - COLS, mask=mask, other=0.0,
                  eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base + COLS, mask=mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base - 1, mask=mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    acc += tl.load(input_ptr + base + 1, mask=mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)

    tl.store(output_ptr + base, acc * 0.25, mask=mask, cache_modifier=".cg")


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

    BLOCK_M = 4
    BLOCK_N = 1024
    PREFIX_BLOCK = 256
    N_BASE = 256

    num_warps = 8
    prefix_num_warps = 4
    boundary_num_warps = 4
    num_stages = 3
    BOUNDARY_BLOCK = 1024

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
            "PREFIX_BLOCK": PREFIX_BLOCK,
            "N_BASE": N_BASE,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "num_warps": num_warps,
            "prefix_num_warps": prefix_num_warps,
            "boundary_num_warps": boundary_num_warps,
            "num_stages": num_stages,
            "aligned_full": 1,
            "store_cg": 1,
            "load_eviction_last": 1,
        })
        return output

    interior_m = rows - 2
    full_m = interior_m // BLOCK_M
    tail_m = interior_m - full_m * BLOCK_M

    interior_end = cols - 1

    prefix_end = N_BASE if interior_end > N_BASE else interior_end
    if prefix_end > 1:
        _jacobi_n_range_kernel[(triton.cdiv(interior_m, BLOCK_M),)](
            input, output,
            ROWS=rows,
            COLS=cols,
            N_START=1,
            N_END=prefix_end,
            BLOCK_M=BLOCK_M,
            BLOCK_N=PREFIX_BLOCK,
            num_warps=prefix_num_warps,
            num_stages=num_stages,
        )

    full_n = 0
    if interior_end > N_BASE:
        full_n = (interior_end - N_BASE) // BLOCK_N

    if full_m > 0 and full_n > 0:
        _jacobi_full_kernel[(full_m, full_n)](
            input, output,
            ROWS=rows,
            COLS=cols,
            N_BASE=N_BASE,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    tail_start = N_BASE + full_n * BLOCK_N
    if tail_start < interior_end:
        _jacobi_n_range_kernel[(triton.cdiv(interior_m, BLOCK_M),)](
            input, output,
            ROWS=rows,
            COLS=cols,
            N_START=tail_start,
            N_END=interior_end,
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
            N_BASE=N_BASE,
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
        "PREFIX_BLOCK": PREFIX_BLOCK,
        "N_BASE": N_BASE,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "num_warps": num_warps,
        "prefix_num_warps": prefix_num_warps,
        "boundary_num_warps": boundary_num_warps,
        "num_stages": num_stages,
        "aligned_full": 1,
        "store_cg": 1,
        "load_eviction_last": 1,
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


@ct.kernel
def _jacobi_full_aligned_kernel(input, output,
                                ROWS: ConstInt, COLS: ConstInt,
                                N_BASE: ConstInt, N_FULL_END: ConstInt,
                                BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    top_arr = input.slice(0, 0, ROWS - 2).slice(1, N_BASE, N_FULL_END)
    bottom_arr = input.slice(0, 2, ROWS).slice(1, N_BASE, N_FULL_END)
    left_arr = input.slice(0, 1, ROWS - 1).slice(1, N_BASE - 1, N_FULL_END - 1)
    right_arr = input.slice(0, 1, ROWS - 1).slice(1, N_BASE + 1, N_FULL_END + 1)
    out_arr = output.slice(0, 1, ROWS - 1).slice(1, N_BASE, N_FULL_END)

    top = ct.astype(
        ct.load(top_arr, index=(bid_m, bid_n), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )
    bottom = ct.astype(
        ct.load(bottom_arr, index=(bid_m, bid_n), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )
    left = ct.astype(
        ct.load(left_arr, index=(bid_m, bid_n), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )
    right = ct.astype(
        ct.load(right_arr, index=(bid_m, bid_n), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )

    avg = (top + bottom + left + right) * 0.25
    ct.store(
        out_arr,
        index=(bid_m, bid_n),
        tile=ct.astype(avg, input.dtype),
        latency=1,
        allow_tma=False,
    )


@ct.kernel
def _jacobi_n_range_kernel(input, output,
                           ROWS: ConstInt, COLS: ConstInt,
                           N_START: ConstInt, N_END: ConstInt,
                           BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)

    top_arr = input.slice(0, 0, ROWS - 2).slice(1, N_START, N_END)
    bottom_arr = input.slice(0, 2, ROWS).slice(1, N_START, N_END)
    left_arr = input.slice(0, 1, ROWS - 1).slice(1, N_START - 1, N_END - 1)
    right_arr = input.slice(0, 1, ROWS - 1).slice(1, N_START + 1, N_END + 1)
    out_arr = output.slice(0, 1, ROWS - 1).slice(1, N_START, N_END)

    top = ct.astype(
        ct.load(top_arr, index=(bid_m, 0), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )
    bottom = ct.astype(
        ct.load(bottom_arr, index=(bid_m, 0), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )
    left = ct.astype(
        ct.load(left_arr, index=(bid_m, 0), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )
    right = ct.astype(
        ct.load(right_arr, index=(bid_m, 0), shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO, latency=1, allow_tma=False),
        np.float32,
    )

    avg = (top + bottom + left + right) * 0.25
    ct.store(
        out_arr,
        index=(bid_m, 0),
        tile=ct.astype(avg, input.dtype),
        latency=1,
        allow_tma=False,
    )


@ct.kernel
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


@ct.kernel
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
    PREFIX_BLOCK = 256
    N_BASE = 256
    BOUNDARY_BLOCK = 1024
    occupancy = 4
    boundary_occupancy = 4

    full_kernel = _jacobi_full_aligned_kernel.with_hints(occupancy=occupancy)
    range_kernel = _jacobi_n_range_kernel.with_hints(occupancy=occupancy)
    boundary_kernel = _boundary_kernel.with_hints(occupancy=boundary_occupancy)
    copy_kernel = _copy_all_kernel.with_hints(occupancy=boundary_occupancy)

    if rows <= 2 or cols <= 2:
        grid = (ct.cdiv(rows, BLOCK_M), ct.cdiv(cols, BLOCK_N), 1)
        ct.launch(stream, grid, copy_kernel, (input, output, rows, cols, BLOCK_M, BLOCK_N))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "PREFIX_BLOCK": PREFIX_BLOCK,
            "N_BASE": N_BASE,
            "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
            "occupancy": occupancy,
            "boundary_occupancy": boundary_occupancy,
            "aligned_sliced": 1,
        })
        return output

    interior_m = rows - 2
    interior_end = cols - 1

    grid_m = ct.cdiv(interior_m, BLOCK_M)

    prefix_end = N_BASE if interior_end > N_BASE else interior_end
    if prefix_end > 1:
        ct.launch(
            stream,
            (grid_m, 1, 1),
            range_kernel,
            (input, output, rows, cols, 1, prefix_end, BLOCK_M, PREFIX_BLOCK),
        )

    full_n = 0
    if interior_end > N_BASE:
        full_n = (interior_end - N_BASE) // BLOCK_N

    if full_n > 0:
        n_full_end = N_BASE + full_n * BLOCK_N
        ct.launch(
            stream,
            (grid_m, full_n, 1),
            full_kernel,
            (input, output, rows, cols, N_BASE, n_full_end, BLOCK_M, BLOCK_N),
        )

    tail_start = N_BASE + full_n * BLOCK_N
    if tail_start < interior_end:
        ct.launch(
            stream,
            (grid_m, 1, 1),
            range_kernel,
            (input, output, rows, cols, tail_start, interior_end, BLOCK_M, BLOCK_N),
        )

    boundary_total = 2 * cols + 2 * (rows - 2)
    ct.launch(
        stream,
        (ct.cdiv(boundary_total, BOUNDARY_BLOCK), 1, 1),
        boundary_kernel,
        (input, output, rows, cols, boundary_total, BOUNDARY_BLOCK),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "PREFIX_BLOCK": PREFIX_BLOCK,
        "N_BASE": N_BASE,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "occupancy": occupancy,
        "boundary_occupancy": boundary_occupancy,
        "aligned_sliced": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
