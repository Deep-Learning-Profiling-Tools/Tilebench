import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _accum_rows8(a0, a1, a2, a3, a4, a5, a6, a7,
                 x0, x1, x2, x3, x4, x5, x6, x7, x8, x9,
                 w0, w1, w2):
    b0 = a0 + x0 * w0 + x1 * w1 + x2 * w2
    b1 = a1 + x1 * w0 + x2 * w1 + x3 * w2
    b2 = a2 + x2 * w0 + x3 * w1 + x4 * w2
    b3 = a3 + x3 * w0 + x4 * w1 + x5 * w2
    b4 = a4 + x4 * w0 + x5 * w1 + x6 * w2
    b5 = a5 + x5 * w0 + x6 * w1 + x7 * w2
    b6 = a6 + x6 * w0 + x7 * w1 + x8 * w2
    b7 = a7 + x7 * w0 + x8 * w1 + x9 * w2
    return b0, b1, b2, b3, b4, b5, b6, b7


@triton.jit
def _conv3d_3x3x3_d4_rows8_kernel(input_ptr, kernel_ptr, output_ptr,
                                  output_depth, output_rows, output_cols,
                                  input_depth, input_rows, input_cols,
                                  BLOCK_COL: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_rb = tl.program_id(1)
    pid_db = tl.program_id(2)

    row0 = pid_rb * 8
    od0 = pid_db * 4

    cols = pid_c * BLOCK_COL + tl.arange(0, BLOCK_COL)
    cmask = cols < output_cols

    in_plane = input_rows * input_cols
    out_plane = output_rows * output_cols

    m0 = cmask & ((row0 + 0) < input_rows)
    m1 = cmask & ((row0 + 1) < input_rows)
    m2 = cmask & ((row0 + 2) < input_rows)
    m3 = cmask & ((row0 + 3) < input_rows)
    m4 = cmask & ((row0 + 4) < input_rows)
    m5 = cmask & ((row0 + 5) < input_rows)
    m6 = cmask & ((row0 + 6) < input_rows)
    m7 = cmask & ((row0 + 7) < input_rows)
    m8 = cmask & ((row0 + 8) < input_rows)
    m9 = cmask & ((row0 + 9) < input_rows)

    acc00 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc02 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc03 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc04 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc05 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc06 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc07 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    acc10 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc13 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc14 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc15 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc16 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc17 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    acc20 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc21 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc22 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc23 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc24 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc25 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc26 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc27 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    acc30 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc31 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc32 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc33 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc34 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc35 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc36 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc37 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    for p in tl.static_range(0, 6):
        depth_ok = (od0 + p) < input_depth
        d_base = (od0 + p) * in_plane + row0 * input_cols + cols

        for kc in tl.static_range(0, 3):
            x0 = tl.load(input_ptr + d_base + 0 * input_cols + kc,
                         mask=m0 & depth_ok, other=0.0).to(tl.float32)
            x1 = tl.load(input_ptr + d_base + 1 * input_cols + kc,
                         mask=m1 & depth_ok, other=0.0).to(tl.float32)
            x2 = tl.load(input_ptr + d_base + 2 * input_cols + kc,
                         mask=m2 & depth_ok, other=0.0).to(tl.float32)
            x3 = tl.load(input_ptr + d_base + 3 * input_cols + kc,
                         mask=m3 & depth_ok, other=0.0).to(tl.float32)
            x4 = tl.load(input_ptr + d_base + 4 * input_cols + kc,
                         mask=m4 & depth_ok, other=0.0).to(tl.float32)
            x5 = tl.load(input_ptr + d_base + 5 * input_cols + kc,
                         mask=m5 & depth_ok, other=0.0).to(tl.float32)
            x6 = tl.load(input_ptr + d_base + 6 * input_cols + kc,
                         mask=m6 & depth_ok, other=0.0).to(tl.float32)
            x7 = tl.load(input_ptr + d_base + 7 * input_cols + kc,
                         mask=m7 & depth_ok, other=0.0).to(tl.float32)
            x8 = tl.load(input_ptr + d_base + 8 * input_cols + kc,
                         mask=m8 & depth_ok, other=0.0).to(tl.float32)
            x9 = tl.load(input_ptr + d_base + 9 * input_cols + kc,
                         mask=m9 & depth_ok, other=0.0).to(tl.float32)

            if p < 3:
                kb0 = p * 9 + kc
                w0 = tl.load(kernel_ptr + kb0 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb0 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb0 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc00, acc01, acc02, acc03, acc04, acc05, acc06, acc07 = _accum_rows8(
                    acc00, acc01, acc02, acc03, acc04, acc05, acc06, acc07,
                    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, w0, w1, w2
                )

            if p >= 1 and p <= 3:
                kb1 = (p - 1) * 9 + kc
                w0 = tl.load(kernel_ptr + kb1 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb1 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb1 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc10, acc11, acc12, acc13, acc14, acc15, acc16, acc17 = _accum_rows8(
                    acc10, acc11, acc12, acc13, acc14, acc15, acc16, acc17,
                    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, w0, w1, w2
                )

            if p >= 2 and p <= 4:
                kb2 = (p - 2) * 9 + kc
                w0 = tl.load(kernel_ptr + kb2 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb2 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb2 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc20, acc21, acc22, acc23, acc24, acc25, acc26, acc27 = _accum_rows8(
                    acc20, acc21, acc22, acc23, acc24, acc25, acc26, acc27,
                    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, w0, w1, w2
                )

            if p >= 3:
                kb3 = (p - 3) * 9 + kc
                w0 = tl.load(kernel_ptr + kb3 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb3 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb3 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc30, acc31, acc32, acc33, acc34, acc35, acc36, acc37 = _accum_rows8(
                    acc30, acc31, acc32, acc33, acc34, acc35, acc36, acc37,
                    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, w0, w1, w2
                )

    dmask0 = (od0 + 0) < output_depth
    base0 = output_ptr + (od0 + 0) * out_plane + row0 * output_cols + cols
    tl.store(base0 + 0 * output_cols, acc00, mask=dmask0 & cmask & ((row0 + 0) < output_rows))
    tl.store(base0 + 1 * output_cols, acc01, mask=dmask0 & cmask & ((row0 + 1) < output_rows))
    tl.store(base0 + 2 * output_cols, acc02, mask=dmask0 & cmask & ((row0 + 2) < output_rows))
    tl.store(base0 + 3 * output_cols, acc03, mask=dmask0 & cmask & ((row0 + 3) < output_rows))
    tl.store(base0 + 4 * output_cols, acc04, mask=dmask0 & cmask & ((row0 + 4) < output_rows))
    tl.store(base0 + 5 * output_cols, acc05, mask=dmask0 & cmask & ((row0 + 5) < output_rows))
    tl.store(base0 + 6 * output_cols, acc06, mask=dmask0 & cmask & ((row0 + 6) < output_rows))
    tl.store(base0 + 7 * output_cols, acc07, mask=dmask0 & cmask & ((row0 + 7) < output_rows))

    dmask1 = (od0 + 1) < output_depth
    base1 = output_ptr + (od0 + 1) * out_plane + row0 * output_cols + cols
    tl.store(base1 + 0 * output_cols, acc10, mask=dmask1 & cmask & ((row0 + 0) < output_rows))
    tl.store(base1 + 1 * output_cols, acc11, mask=dmask1 & cmask & ((row0 + 1) < output_rows))
    tl.store(base1 + 2 * output_cols, acc12, mask=dmask1 & cmask & ((row0 + 2) < output_rows))
    tl.store(base1 + 3 * output_cols, acc13, mask=dmask1 & cmask & ((row0 + 3) < output_rows))
    tl.store(base1 + 4 * output_cols, acc14, mask=dmask1 & cmask & ((row0 + 4) < output_rows))
    tl.store(base1 + 5 * output_cols, acc15, mask=dmask1 & cmask & ((row0 + 5) < output_rows))
    tl.store(base1 + 6 * output_cols, acc16, mask=dmask1 & cmask & ((row0 + 6) < output_rows))
    tl.store(base1 + 7 * output_cols, acc17, mask=dmask1 & cmask & ((row0 + 7) < output_rows))

    dmask2 = (od0 + 2) < output_depth
    base2 = output_ptr + (od0 + 2) * out_plane + row0 * output_cols + cols
    tl.store(base2 + 0 * output_cols, acc20, mask=dmask2 & cmask & ((row0 + 0) < output_rows))
    tl.store(base2 + 1 * output_cols, acc21, mask=dmask2 & cmask & ((row0 + 1) < output_rows))
    tl.store(base2 + 2 * output_cols, acc22, mask=dmask2 & cmask & ((row0 + 2) < output_rows))
    tl.store(base2 + 3 * output_cols, acc23, mask=dmask2 & cmask & ((row0 + 3) < output_rows))
    tl.store(base2 + 4 * output_cols, acc24, mask=dmask2 & cmask & ((row0 + 4) < output_rows))
    tl.store(base2 + 5 * output_cols, acc25, mask=dmask2 & cmask & ((row0 + 5) < output_rows))
    tl.store(base2 + 6 * output_cols, acc26, mask=dmask2 & cmask & ((row0 + 6) < output_rows))
    tl.store(base2 + 7 * output_cols, acc27, mask=dmask2 & cmask & ((row0 + 7) < output_rows))

    dmask3 = (od0 + 3) < output_depth
    base3 = output_ptr + (od0 + 3) * out_plane + row0 * output_cols + cols
    tl.store(base3 + 0 * output_cols, acc30, mask=dmask3 & cmask & ((row0 + 0) < output_rows))
    tl.store(base3 + 1 * output_cols, acc31, mask=dmask3 & cmask & ((row0 + 1) < output_rows))
    tl.store(base3 + 2 * output_cols, acc32, mask=dmask3 & cmask & ((row0 + 2) < output_rows))
    tl.store(base3 + 3 * output_cols, acc33, mask=dmask3 & cmask & ((row0 + 3) < output_rows))
    tl.store(base3 + 4 * output_cols, acc34, mask=dmask3 & cmask & ((row0 + 4) < output_rows))
    tl.store(base3 + 5 * output_cols, acc35, mask=dmask3 & cmask & ((row0 + 5) < output_rows))
    tl.store(base3 + 6 * output_cols, acc36, mask=dmask3 & cmask & ((row0 + 6) < output_rows))
    tl.store(base3 + 7 * output_cols, acc37, mask=dmask3 & cmask & ((row0 + 7) < output_rows))


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

    num_warps = 4
    num_stages = 3

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        BLOCK_COL = 64
        ROWS_PER_BLOCK = 8
        DEPTHS_PER_BLOCK = 4
        grid = (
            triton.cdiv(output_cols, BLOCK_COL),
            triton.cdiv(output_rows, ROWS_PER_BLOCK),
            triton.cdiv(output_depth, DEPTHS_PER_BLOCK),
        )
        _conv3d_3x3x3_d4_rows8_kernel[grid](
            input, kernel, output,
            output_depth, output_rows, output_cols,
            input_depth, input_rows, input_cols,
            BLOCK_COL=BLOCK_COL,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_COL": BLOCK_COL,
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
            "DEPTHS_PER_BLOCK": DEPTHS_PER_BLOCK,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "variant": "3x3x3_d4_rows8",
        })
    else:
        BLOCK_COL = 256
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
