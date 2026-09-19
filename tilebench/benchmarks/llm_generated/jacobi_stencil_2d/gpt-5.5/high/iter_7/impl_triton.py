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
