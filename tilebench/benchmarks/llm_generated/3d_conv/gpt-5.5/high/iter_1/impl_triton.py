import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv3d_3x3x3_row_mma_kernel(input_ptr, kernel_ptr, output_ptr,
                                 output_rows, output_cols, input_rows, input_cols,
                                 BLOCK_R: tl.constexpr,
                                 BLOCK_COL: tl.constexpr,
                                 BLOCK_K: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_r = tl.program_id(1)
    od = tl.program_id(2)

    row_start = pid_r * BLOCK_R
    col_start = pid_c * BLOCK_COL

    r = tl.arange(0, BLOCK_R)
    k = tl.arange(0, BLOCK_K)
    cols = col_start + tl.arange(0, BLOCK_COL)

    rel = k[None, :] - r[:, None]

    in_plane = input_rows * input_cols
    out_plane = output_rows * output_cols

    acc = tl.zeros((BLOCK_R, BLOCK_COL), dtype=tl.float32)

    for kd in tl.static_range(0, 3):
        for kc in tl.static_range(0, 3):
            x_rows = row_start + k
            x_ptrs = (
                input_ptr
                + (od + kd) * in_plane
                + x_rows[:, None] * input_cols
                + (cols[None, :] + kc)
            )
            x_mask = (
                (k[:, None] < (BLOCK_R + 2))
                & (x_rows[:, None] < input_rows)
                & (cols[None, :] < output_cols)
            )
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            w0 = tl.load(kernel_ptr + kd * 9 + 0 * 3 + kc,
                         eviction_policy="evict_last").to(tl.float32)
            w1 = tl.load(kernel_ptr + kd * 9 + 1 * 3 + kc,
                         eviction_policy="evict_last").to(tl.float32)
            w2 = tl.load(kernel_ptr + kd * 9 + 2 * 3 + kc,
                         eviction_policy="evict_last").to(tl.float32)

            w = tl.zeros((BLOCK_R, BLOCK_K), dtype=tl.float32)
            w = tl.where(rel == 0, w0, w)
            w = tl.where(rel == 1, w1, w)
            w = tl.where(rel == 2, w2, w)

            acc = tl.dot(w, x, acc, input_precision="tf32")

    out_rows = row_start + r
    out_ptrs = (
        output_ptr
        + od * out_plane
        + out_rows[:, None] * output_cols
        + cols[None, :]
    )
    out_mask = (out_rows[:, None] < output_rows) & (cols[None, :] < output_cols)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _conv3d_generic_kernel(input_ptr, kernel_ptr, output_ptr,
                           output_rows, output_cols, input_rows, input_cols,
                           BLOCK_COL: tl.constexpr,
                           KERNEL_DEPTH: tl.constexpr,
                           KERNEL_ROWS: tl.constexpr,
                           KERNEL_COLS: tl.constexpr):
    pid_c = tl.program_id(0)
    orow = tl.program_id(1)
    od = tl.program_id(2)

    cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)
    mask = cols < output_cols

    in_plane = input_rows * input_cols
    out_plane = output_rows * output_cols

    acc = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    for kd in tl.static_range(0, KERNEL_DEPTH):
        for kr in tl.static_range(0, KERNEL_ROWS):
            base = (od + kd) * in_plane + (orow + kr) * input_cols + cols
            k_base = (kd * KERNEL_ROWS + kr) * KERNEL_COLS
            for kc in tl.static_range(0, KERNEL_COLS):
                x = tl.load(input_ptr + base + kc, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(kernel_ptr + k_base + kc,
                            eviction_policy="evict_last").to(tl.float32)
                acc += x * w

    out = od * out_plane + orow * output_cols + cols
    tl.store(output_ptr + out, acc, mask=mask)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    output = torch.empty((output_depth, output_rows, output_cols),
                         device=input.device, dtype=input.dtype)

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        BLOCK_R = 16
        BLOCK_COL = 128
        BLOCK_K = 32
        num_warps = 4
        num_stages = 3

        grid = (
            triton.cdiv(output_cols, BLOCK_COL),
            triton.cdiv(output_rows, BLOCK_R),
            output_depth,
        )
        _conv3d_3x3x3_row_mma_kernel[grid](
            input, kernel, output,
            output_rows, output_cols, input_rows, input_cols,
            BLOCK_R=BLOCK_R,
            BLOCK_COL=BLOCK_COL,
            BLOCK_K=BLOCK_K,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_R": BLOCK_R,
            "BLOCK_COL": BLOCK_COL,
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "variant": "3x3x3_row_mma_sparse",
        })
    else:
        BLOCK_COL = 256
        num_warps = 4
        num_stages = 3

        grid = (triton.cdiv(output_cols, BLOCK_COL), output_rows, output_depth)
        _conv3d_generic_kernel[grid](
            input, kernel, output,
            output_rows, output_cols, input_rows, input_cols,
            BLOCK_COL=BLOCK_COL,
            KERNEL_DEPTH=kernel_depth,
            KERNEL_ROWS=kernel_rows,
            KERNEL_COLS=kernel_cols,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_COL": BLOCK_COL,
            "ROWS_PER_BLOCK": 1,
            "KERNEL_DEPTH": kernel_depth,
            "KERNEL_ROWS": kernel_rows,
            "KERNEL_COLS": kernel_cols,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "variant": "generic",
        })

    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
