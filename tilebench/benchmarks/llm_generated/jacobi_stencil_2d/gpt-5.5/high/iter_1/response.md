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

    offs_m = 1 + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = 1 + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = offs_m[:, None]
    c = offs_n[None, :]
    base = r * COLS + c

    acc = tl.load(input_ptr + base - COLS).to(tl.float32)
    acc += tl.load(input_ptr + base + COLS).to(tl.float32)
    acc += tl.load(input_ptr + base - 1).to(tl.float32)
    acc += tl.load(input_ptr + base + 1).to(tl.float32)

    tl.store(output_ptr + base, acc * 0.25)


@triton.jit
def _jacobi_right_tail_kernel(input_ptr, output_ptr,
                              ROWS: tl.constexpr, COLS: tl.constexpr,
                              N_START: tl.constexpr,
                              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)

    offs_m = 1 + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = N_START + tl.arange(0, BLOCK_N)

    r = offs_m[:, None]
    c = offs_n[None, :]
    base = r * COLS + c
    mask = (r < ROWS - 1) & (c < COLS - 1)

    acc = tl.load(input_ptr + base - COLS, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + base + COLS, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + base - 1, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)

    tl.store(output_ptr + base, acc * 0.25, mask=mask)


@triton.jit
def _jacobi_bottom_tail_kernel(input_ptr, output_ptr,
                               ROWS: tl.constexpr, COLS: tl.constexpr,
                               M_START: tl.constexpr,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)

    offs_m = M_START + tl.arange(0, BLOCK_M)
    offs_n = 1 + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    r = offs_m[:, None]
    c = offs_n[None, :]
    base = r * COLS + c
    mask = (r < ROWS - 1) & (c < COLS - 1)

    acc = tl.load(input_ptr + base - COLS, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + base + COLS, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + base - 1, mask=mask, other=0.0).to(tl.float32)
    acc += tl.load(input_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)

    tl.store(output_ptr + base, acc * 0.25, mask=mask)


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
        grid = (triton.cdiv(n_elements, BOUNDARY_BLOCK),)
        _copy_all_kernel[grid](
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
def _jacobi_kernel(input, output,
                   ROWS: ConstInt, COLS: ConstInt,
                   BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)[:, None]
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)[None, :]

    r = bid_m * BLOCK_M + offs_m
    c = bid_n * BLOCK_N + offs_n

    center = ct.astype(
        ct.load(
            input,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )

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

    interior = (r > 0) & (r < ROWS - 1) & (c > 0) & (c < COLS - 1)
    avg = (top + bottom + left + right) * 0.25
    result = ct.where(interior, avg, center)

    ct.store(
        output,
        index=(bid_m, bid_n),
        tile=ct.astype(result, input.dtype),
        latency=1,
        allow_tma=False,
    )


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_M = 4
    BLOCK_N = 512
    occupancy = 4

    grid = (ct.cdiv(rows, BLOCK_M), ct.cdiv(cols, BLOCK_N), 1)
    ct.launch(stream, grid, _jacobi_kernel, (input, output, rows, cols, BLOCK_M, BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
