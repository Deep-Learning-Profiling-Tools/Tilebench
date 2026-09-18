import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_stripe_kernel(input_ptr, output_ptr,
                          ROWS: tl.constexpr, COLS: tl.constexpr,
                          OUT_M: tl.constexpr, LOAD_N: tl.constexpr,
                          OUT_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs = tl.arange(0, LOAD_N)
    load_c = pid_n * OUT_N + offs
    col_load_mask = load_c < COLS

    idx_center = tl.minimum(offs + 1, LOAD_N - 1)
    idx_right = tl.minimum(offs + 2, LOAD_N - 1)

    base_r = pid_m * OUT_M

    r0 = base_r
    r1 = base_r + 1

    row0 = tl.load(
        input_ptr + r0 * COLS + load_c,
        mask=(r0 < ROWS) & col_load_mask,
        other=0.0,
    ).to(tl.float32)
    row1 = tl.load(
        input_ptr + r1 * COLS + load_c,
        mask=(r1 < ROWS) & col_load_mask,
        other=0.0,
    ).to(tl.float32)

    out_c = pid_n * OUT_N + 1 + offs
    col_store_mask = (offs < OUT_N) & (out_c < COLS - 1)

    for i in tl.static_range(0, OUT_M):
        r2 = base_r + i + 2
        row2 = tl.load(
            input_ptr + r2 * COLS + load_c,
            mask=(r2 < ROWS) & col_load_mask,
            other=0.0,
        ).to(tl.float32)

        top = tl.gather(row0, idx_center, 0)
        bottom = tl.gather(row2, idx_center, 0)
        left = row1
        right = tl.gather(row1, idx_right, 0)

        out_r = base_r + i + 1
        store_mask = col_store_mask & (out_r < ROWS - 1)

        acc = top + bottom
        acc += left
        acc += right

        tl.store(
            output_ptr + out_r * COLS + out_c,
            acc * 0.25,
            mask=store_mask,
        )

        row0 = row1
        row1 = row2


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

    OUT_M = 16
    LOAD_N = 1024
    OUT_N = 1022
    BOUNDARY_BLOCK = 1024
    num_warps = 8
    boundary_num_warps = 4
    num_stages = 3

    if rows <= 2 or cols <= 2:
        n_elements = rows * cols
        _copy_all_kernel[(triton.cdiv(n_elements, BOUNDARY_BLOCK),)](
            input, output,
            N=n_elements,
            BLOCK=BOUNDARY_BLOCK,
            num_warps=boundary_num_warps,
            num_stages=num_stages,
        )
    else:
        interior_m = rows - 2
        interior_n = cols - 2
        grid = (triton.cdiv(interior_m, OUT_M), triton.cdiv(interior_n, OUT_N))
        _jacobi_stripe_kernel[grid](
            input, output,
            ROWS=rows,
            COLS=cols,
            OUT_M=OUT_M,
            LOAD_N=LOAD_N,
            OUT_N=OUT_N,
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
        "LOAD_N": LOAD_N,
        "OUT_N": OUT_N,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "num_warps": num_warps,
        "boundary_num_warps": boundary_num_warps,
        "num_stages": num_stages,
        "stripe_halo": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
