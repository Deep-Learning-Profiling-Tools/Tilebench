import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_halo_kernel(input_ptr, output_ptr,
                        ROWS: tl.constexpr, COLS: tl.constexpr,
                        LOAD_M: tl.constexpr, LOAD_N: tl.constexpr,
                        OUT_M: tl.constexpr, OUT_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = tl.arange(0, LOAD_M)
    offs_n = tl.arange(0, LOAD_N)

    in_r = pid_m * OUT_M + offs_m[:, None]
    in_c = pid_n * OUT_N + offs_n[None, :]
    load_mask = (in_r < ROWS) & (in_c < COLS)

    tile = tl.load(
        input_ptr + in_r * COLS + in_c,
        mask=load_mask,
        other=0.0,
    ).to(tl.float32)

    row_center = tl.minimum(offs_m + 1, LOAD_M - 1)
    row_bottom = tl.minimum(offs_m + 2, LOAD_M - 1)
    col_center = tl.minimum(offs_n + 1, LOAD_N - 1)
    col_right = tl.minimum(offs_n + 2, LOAD_N - 1)

    top = tl.gather(tile, col_center, 1)
    bottom_rows = tl.gather(tile, row_bottom, 0)
    bottom = tl.gather(bottom_rows, col_center, 1)
    center_rows = tl.gather(tile, row_center, 0)
    left = center_rows
    right = tl.gather(center_rows, col_right, 1)

    out_r = pid_m * OUT_M + 1 + offs_m[:, None]
    out_c = pid_n * OUT_N + 1 + offs_n[None, :]
    store_mask = (
        (offs_m[:, None] < OUT_M)
        & (offs_n[None, :] < OUT_N)
        & (out_r < ROWS - 1)
        & (out_c < COLS - 1)
    )

    tl.store(output_ptr + out_r * COLS + out_c, (top + bottom + left + right) * 0.25, mask=store_mask)


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

    LOAD_M = 16
    LOAD_N = 512
    OUT_M = 14
    OUT_N = 510
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
        _jacobi_halo_kernel[grid](
            input, output,
            ROWS=rows,
            COLS=cols,
            LOAD_M=LOAD_M,
            LOAD_N=LOAD_N,
            OUT_M=OUT_M,
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
        "LOAD_M": LOAD_M,
        "LOAD_N": LOAD_N,
        "OUT_M": OUT_M,
        "OUT_N": OUT_N,
        "BOUNDARY_BLOCK": BOUNDARY_BLOCK,
        "num_warps": num_warps,
        "boundary_num_warps": boundary_num_warps,
        "num_stages": num_stages,
        "halo_tile": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
