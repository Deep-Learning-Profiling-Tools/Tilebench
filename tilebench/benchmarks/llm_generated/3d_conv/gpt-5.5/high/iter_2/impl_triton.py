import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv3d_3x3x3_rows4_kernel(input_ptr, kernel_ptr, output_ptr,
                               output_rows, output_cols, input_rows, input_cols,
                               BLOCK_COL: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_rb = tl.program_id(1)
    od = tl.program_id(2)

    row0 = pid_rb * 4
    cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)
    cmask = cols < output_cols

    in_plane = input_rows * input_cols
    out_plane = output_rows * output_cols

    acc0 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    m0 = cmask & ((row0 + 0) < input_rows)
    m1 = cmask & ((row0 + 1) < input_rows)
    m2 = cmask & ((row0 + 2) < input_rows)
    m3 = cmask & ((row0 + 3) < input_rows)
    m4 = cmask & ((row0 + 4) < input_rows)
    m5 = cmask & ((row0 + 5) < input_rows)

    for kd in tl.static_range(0, 3):
        d_base = (od + kd) * in_plane + row0 * input_cols + cols
        k_base = kd * 9
        for kc in tl.static_range(0, 3):
            x0 = tl.load(input_ptr + d_base + 0 * input_cols + kc,
                         mask=m0, other=0.0).to(tl.float32)
            x1 = tl.load(input_ptr + d_base + 1 * input_cols + kc,
                         mask=m1, other=0.0).to(tl.float32)
            x2 = tl.load(input_ptr + d_base + 2 * input_cols + kc,
                         mask=m2, other=0.0).to(tl.float32)
            x3 = tl.load(input_ptr + d_base + 3 * input_cols + kc,
                         mask=m3, other=0.0).to(tl.float32)
            x4 = tl.load(input_ptr + d_base + 4 * input_cols + kc,
                         mask=m4, other=0.0).to(tl.float32)
            x5 = tl.load(input_ptr + d_base + 5 * input_cols + kc,
                         mask=m5, other=0.0).to(tl.float32)

            w0 = tl.load(kernel_ptr + k_base + 0 * 3 + kc,
                         eviction_policy="evict_last").to(tl.float32)
            w1 = tl.load(kernel_ptr + k_base + 1 * 3 + kc,
                         eviction_policy="evict_last").to(tl.float32)
            w2 = tl.load(kernel_ptr + k_base + 2 * 3 + kc,
                         eviction_policy="evict_last").to(tl.float32)

            acc0 += x0 * w0 + x1 * w1 + x2 * w2
            acc1 += x1 * w0 + x2 * w1 + x3 * w2
            acc2 += x2 * w0 + x3 * w1 + x4 * w2
            acc3 += x3 * w0 + x4 * w1 + x5 * w2

    out_base = output_ptr + od * out_plane + row0 * output_cols + cols
    tl.store(out_base + 0 * output_cols, acc0,
             mask=cmask & ((row0 + 0) < output_rows))
    tl.store(out_base + 1 * output_cols, acc1,
             mask=cmask & ((row0 + 1) < output_rows))
    tl.store(out_base + 2 * output_cols, acc2,
             mask=cmask & ((row0 + 2) < output_rows))
    tl.store(out_base + 3 * output_cols, acc3,
             mask=cmask & ((row0 + 3) < output_rows))


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

    BLOCK_COL = 256
    num_warps = 4
    num_stages = 3

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        ROWS_PER_BLOCK = 4
        grid = (
            triton.cdiv(output_cols, BLOCK_COL),
            triton.cdiv(output_rows, ROWS_PER_BLOCK),
            output_depth,
        )
        _conv3d_3x3x3_rows4_kernel[grid](
            input, kernel, output,
            output_rows, output_cols, input_rows, input_cols,
            BLOCK_COL=BLOCK_COL,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_COL": BLOCK_COL,
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "variant": "3x3x3_rows4",
        })
    else:
        ROWS_PER_BLOCK = 1
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
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
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
