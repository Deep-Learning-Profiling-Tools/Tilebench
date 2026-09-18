```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _accum_rows4(a0, a1, a2, a3, x0, x1, x2, x3, x4, x5, w0, w1, w2):
    b0 = a0 + x0 * w0 + x1 * w1 + x2 * w2
    b1 = a1 + x1 * w0 + x2 * w1 + x3 * w2
    b2 = a2 + x2 * w0 + x3 * w1 + x4 * w2
    b3 = a3 + x3 * w0 + x4 * w1 + x5 * w2
    return b0, b1, b2, b3


@triton.jit
def _conv3d_3x3x3_d5_rows4_kernel(input_ptr, kernel_ptr, output_ptr,
                                  output_depth, output_rows, output_cols,
                                  input_depth, input_rows, input_cols,
                                  BLOCK_COL: tl.constexpr):
    pid_c = tl.program_id(0)
    pid_rb = tl.program_id(1)
    pid_db = tl.program_id(2)

    row0 = pid_rb * 4
    od0 = pid_db * 5

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

    acc00 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc02 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc03 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    acc10 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc13 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    acc20 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc21 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc22 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc23 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    acc30 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc31 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc32 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc33 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    acc40 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc41 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc42 = tl.zeros((BLOCK_COL,), dtype=tl.float32)
    acc43 = tl.zeros((BLOCK_COL,), dtype=tl.float32)

    for p in tl.static_range(0, 7):
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

            if p < 3:
                kb0 = p * 9 + kc
                w0 = tl.load(kernel_ptr + kb0 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb0 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb0 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc00, acc01, acc02, acc03 = _accum_rows4(
                    acc00, acc01, acc02, acc03, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

            if p >= 1 and p <= 3:
                kb1 = (p - 1) * 9 + kc
                w0 = tl.load(kernel_ptr + kb1 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb1 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb1 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc10, acc11, acc12, acc13 = _accum_rows4(
                    acc10, acc11, acc12, acc13, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

            if p >= 2 and p <= 4:
                kb2 = (p - 2) * 9 + kc
                w0 = tl.load(kernel_ptr + kb2 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb2 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb2 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc20, acc21, acc22, acc23 = _accum_rows4(
                    acc20, acc21, acc22, acc23, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

            if p >= 3 and p <= 5:
                kb3 = (p - 3) * 9 + kc
                w0 = tl.load(kernel_ptr + kb3 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb3 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb3 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc30, acc31, acc32, acc33 = _accum_rows4(
                    acc30, acc31, acc32, acc33, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

            if p >= 4:
                kb4 = (p - 4) * 9 + kc
                w0 = tl.load(kernel_ptr + kb4 + 0 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w1 = tl.load(kernel_ptr + kb4 + 1 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                w2 = tl.load(kernel_ptr + kb4 + 2 * 3,
                             eviction_policy="evict_last").to(tl.float32)
                acc40, acc41, acc42, acc43 = _accum_rows4(
                    acc40, acc41, acc42, acc43, x0, x1, x2, x3, x4, x5, w0, w1, w2
                )

    base0 = output_ptr + (od0 + 0) * out_plane + row0 * output_cols + cols
    dmask0 = (od0 + 0) < output_depth
    tl.store(base0 + 0 * output_cols, acc00,
             mask=dmask0 & cmask & ((row0 + 0) < output_rows))
    tl.store(base0 + 1 * output_cols, acc01,
             mask=dmask0 & cmask & ((row0 + 1) < output_rows))
    tl.store(base0 + 2 * output_cols, acc02,
             mask=dmask0 & cmask & ((row0 + 2) < output_rows))
    tl.store(base0 + 3 * output_cols, acc03,
             mask=dmask0 & cmask & ((row0 + 3) < output_rows))

    base1 = output_ptr + (od0 + 1) * out_plane + row0 * output_cols + cols
    dmask1 = (od0 + 1) < output_depth
    tl.store(base1 + 0 * output_cols, acc10,
             mask=dmask1 & cmask & ((row0 + 0) < output_rows))
    tl.store(base1 + 1 * output_cols, acc11,
             mask=dmask1 & cmask & ((row0 + 1) < output_rows))
    tl.store(base1 + 2 * output_cols, acc12,
             mask=dmask1 & cmask & ((row0 + 2) < output_rows))
    tl.store(base1 + 3 * output_cols, acc13,
             mask=dmask1 & cmask & ((row0 + 3) < output_rows))

    base2 = output_ptr + (od0 + 2) * out_plane + row0 * output_cols + cols
    dmask2 = (od0 + 2) < output_depth
    tl.store(base2 + 0 * output_cols, acc20,
             mask=dmask2 & cmask & ((row0 + 0) < output_rows))
    tl.store(base2 + 1 * output_cols, acc21,
             mask=dmask2 & cmask & ((row0 + 1) < output_rows))
    tl.store(base2 + 2 * output_cols, acc22,
             mask=dmask2 & cmask & ((row0 + 2) < output_rows))
    tl.store(base2 + 3 * output_cols, acc23,
             mask=dmask2 & cmask & ((row0 + 3) < output_rows))

    base3 = output_ptr + (od0 + 3) * out_plane + row0 * output_cols + cols
    dmask3 = (od0 + 3) < output_depth
    tl.store(base3 + 0 * output_cols, acc30,
             mask=dmask3 & cmask & ((row0 + 0) < output_rows))
    tl.store(base3 + 1 * output_cols, acc31,
             mask=dmask3 & cmask & ((row0 + 1) < output_rows))
    tl.store(base3 + 2 * output_cols, acc32,
             mask=dmask3 & cmask & ((row0 + 2) < output_rows))
    tl.store(base3 + 3 * output_cols, acc33,
             mask=dmask3 & cmask & ((row0 + 3) < output_rows))

    base4 = output_ptr + (od0 + 4) * out_plane + row0 * output_cols + cols
    dmask4 = (od0 + 4) < output_depth
    tl.store(base4 + 0 * output_cols, acc40,
             mask=dmask4 & cmask & ((row0 + 0) < output_rows))
    tl.store(base4 + 1 * output_cols, acc41,
             mask=dmask4 & cmask & ((row0 + 1) < output_rows))
    tl.store(base4 + 2 * output_cols, acc42,
             mask=dmask4 & cmask & ((row0 + 2) < output_rows))
    tl.store(base4 + 3 * output_cols, acc43,
             mask=dmask4 & cmask & ((row0 + 3) < output_rows))


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
        BLOCK_COL = 128
        ROWS_PER_BLOCK = 4
        DEPTHS_PER_BLOCK = 5
        grid = (
            triton.cdiv(output_cols, BLOCK_COL),
            triton.cdiv(output_rows, ROWS_PER_BLOCK),
            triton.cdiv(output_depth, DEPTHS_PER_BLOCK),
        )
        _conv3d_3x3x3_d5_rows4_kernel[grid](
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
            "variant": "3x3x3_d5_rows4",
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.function
def _accum_rows4(a0, a1, a2, a3, x0, x1, x2, x3, x4, x5, w0, w1, w2):
    b0 = a0 + x0 * w0 + x1 * w1 + x2 * w2
    b1 = a1 + x1 * w0 + x2 * w1 + x3 * w2
    b2 = a2 + x2 * w0 + x3 * w1 + x4 * w2
    b3 = a3 + x3 * w0 + x4 * w1 + x5 * w2
    return b0, b1, b2, b3


@ct.kernel(occupancy=4)
def _conv3d_3x3x3_d5_rows4_kc_outer_kernel(input, kernel, output,
                                           ROWS_PER_BLOCK: ConstInt,
                                           DEPTHS_PER_BLOCK: ConstInt,
                                           TILE: ConstInt):
    bid_c = ct.bid(0)
    bid_rb = ct.bid(1)
    bid_db = ct.bid(2)

    row0 = bid_rb * ROWS_PER_BLOCK
    od0 = bid_db * DEPTHS_PER_BLOCK

    acc00 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc01 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc02 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc03 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    acc10 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc11 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc12 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc13 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    acc20 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc21 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc22 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc23 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    acc30 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc31 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc32 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc33 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    acc40 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc41 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc42 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)
    acc43 = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    for kc in range(0, 3):
        shifted = input.slice(2, kc, input.shape[2])

        w000 = ct.astype(ct.load(kernel, index=(0, 0, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w010 = ct.astype(ct.load(kernel, index=(0, 1, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w020 = ct.astype(ct.load(kernel, index=(0, 2, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w100 = ct.astype(ct.load(kernel, index=(1, 0, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w110 = ct.astype(ct.load(kernel, index=(1, 1, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w120 = ct.astype(ct.load(kernel, index=(1, 2, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w200 = ct.astype(ct.load(kernel, index=(2, 0, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w210 = ct.astype(ct.load(kernel, index=(2, 1, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)
        w220 = ct.astype(ct.load(kernel, index=(2, 2, kc), shape=(1, 1, 1),
                                 padding_mode=ct.PaddingMode.ZERO,
                                 allow_tma=False, latency=1), np.float32)

        for p in range(0, 7):
            x0 = ct.astype(ct.load(
                shifted, index=(od0 + p, row0 + 0, bid_c),
                shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
                latency=1,
            ), np.float32)
            x1 = ct.astype(ct.load(
                shifted, index=(od0 + p, row0 + 1, bid_c),
                shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
                latency=1,
            ), np.float32)
            x2 = ct.astype(ct.load(
                shifted, index=(od0 + p, row0 + 2, bid_c),
                shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
                latency=1,
            ), np.float32)
            x3 = ct.astype(ct.load(
                shifted, index=(od0 + p, row0 + 3, bid_c),
                shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
                latency=1,
            ), np.float32)
            x4 = ct.astype(ct.load(
                shifted, index=(od0 + p, row0 + 4, bid_c),
                shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
                latency=1,
            ), np.float32)
            x5 = ct.astype(ct.load(
                shifted, index=(od0 + p, row0 + 5, bid_c),
                shape=(1, 1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
                latency=1,
            ), np.float32)

            if p < 3:
                if p == 0:
                    ww0 = w000
                    ww1 = w010
                    ww2 = w020
                elif p == 1:
                    ww0 = w100
                    ww1 = w110
                    ww2 = w120
                else:
                    ww0 = w200
                    ww1 = w210
                    ww2 = w220
                acc00, acc01, acc02, acc03 = _accum_rows4(
                    acc00, acc01, acc02, acc03, x0, x1, x2, x3, x4, x5, ww0, ww1, ww2
                )

            if p >= 1 and p <= 3:
                if p == 1:
                    ww0 = w000
                    ww1 = w010
                    ww2 = w020
                elif p == 2:
                    ww0 = w100
                    ww1 = w110
                    ww2 = w120
                else:
                    ww0 = w200
                    ww1 = w210
                    ww2 = w220
                acc10, acc11, acc12, acc13 = _accum_rows4(
                    acc10, acc11, acc12, acc13, x0, x1, x2, x3, x4, x5, ww0, ww1, ww2
                )

            if p >= 2 and p <= 4:
                if p == 2:
                    ww0 = w000
                    ww1 = w010
                    ww2 = w020
                elif p == 3:
                    ww0 = w100
                    ww1 = w110
                    ww2 = w120
                else:
                    ww0 = w200
                    ww1 = w210
                    ww2 = w220
                acc20, acc21, acc22, acc23 = _accum_rows4(
                    acc20, acc21, acc22, acc23, x0, x1, x2, x3, x4, x5, ww0, ww1, ww2
                )

            if p >= 3 and p <= 5:
                if p == 3:
                    ww0 = w000
                    ww1 = w010
                    ww2 = w020
                elif p == 4:
                    ww0 = w100
                    ww1 = w110
                    ww2 = w120
                else:
                    ww0 = w200
                    ww1 = w210
                    ww2 = w220
                acc30, acc31, acc32, acc33 = _accum_rows4(
                    acc30, acc31, acc32, acc33, x0, x1, x2, x3, x4, x5, ww0, ww1, ww2
                )

            if p >= 4:
                if p == 4:
                    ww0 = w000
                    ww1 = w010
                    ww2 = w020
                elif p == 5:
                    ww0 = w100
                    ww1 = w110
                    ww2 = w120
                else:
                    ww0 = w200
                    ww1 = w210
                    ww2 = w220
                acc40, acc41, acc42, acc43 = _accum_rows4(
                    acc40, acc41, acc42, acc43, x0, x1, x2, x3, x4, x5, ww0, ww1, ww2
                )

    ct.store(output, index=(od0 + 0, row0 + 0, bid_c),
             tile=ct.astype(acc00, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 0, row0 + 1, bid_c),
             tile=ct.astype(acc01, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 0, row0 + 2, bid_c),
             tile=ct.astype(acc02, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 0, row0 + 3, bid_c),
             tile=ct.astype(acc03, output.dtype), allow_tma=False, latency=1)

    ct.store(output, index=(od0 + 1, row0 + 0, bid_c),
             tile=ct.astype(acc10, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 1, row0 + 1, bid_c),
             tile=ct.astype(acc11, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 1, row0 + 2, bid_c),
             tile=ct.astype(acc12, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 1, row0 + 3, bid_c),
             tile=ct.astype(acc13, output.dtype), allow_tma=False, latency=1)

    ct.store(output, index=(od0 + 2, row0 + 0, bid_c),
             tile=ct.astype(acc20, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 2, row0 + 1, bid_c),
             tile=ct.astype(acc21, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 2, row0 + 2, bid_c),
             tile=ct.astype(acc22, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 2, row0 + 3, bid_c),
             tile=ct.astype(acc23, output.dtype), allow_tma=False, latency=1)

    ct.store(output, index=(od0 + 3, row0 + 0, bid_c),
             tile=ct.astype(acc30, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 3, row0 + 1, bid_c),
             tile=ct.astype(acc31, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 3, row0 + 2, bid_c),
             tile=ct.astype(acc32, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 3, row0 + 3, bid_c),
             tile=ct.astype(acc33, output.dtype), allow_tma=False, latency=1)

    ct.store(output, index=(od0 + 4, row0 + 0, bid_c),
             tile=ct.astype(acc40, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 4, row0 + 1, bid_c),
             tile=ct.astype(acc41, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 4, row0 + 2, bid_c),
             tile=ct.astype(acc42, output.dtype), allow_tma=False, latency=1)
    ct.store(output, index=(od0 + 4, row0 + 3, bid_c),
             tile=ct.astype(acc43, output.dtype), allow_tma=False, latency=1)


@ct.kernel(occupancy=4)
def _conv3d_generic_kernel(input, kernel, output,
                           KERNEL_DEPTH: ConstInt,
                           KERNEL_ROWS: ConstInt,
                           KERNEL_COLS: ConstInt,
                           TILE: ConstInt):
    bid_c = ct.bid(0)
    orow = ct.bid(1)
    od = ct.bid(2)

    acc = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    for kd in range(0, KERNEL_DEPTH):
        for kr in range(0, KERNEL_ROWS):
            for kc in range(0, KERNEL_COLS):
                shifted = input.slice(2, kc, input.shape[2])
                x = ct.astype(
                    ct.load(
                        shifted, index=(od + kd, orow + kr, bid_c),
                        shape=(1, 1, TILE),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                w = ct.astype(
                    ct.load(
                        kernel, index=(kd, kr, kc), shape=(1, 1, 1),
                        padding_mode=ct.PaddingMode.ZERO,
                        allow_tma=False,
                        latency=1,
                    ),
                    np.float32,
                )
                acc = acc + x * w

    ct.store(
        output,
        index=(od, orow, bid_c),
        tile=ct.astype(acc, output.dtype),
        allow_tma=False,
        latency=1,
    )


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1

    input_3d = input.view(input_depth, input_rows, input_cols)
    kernel_3d = kernel.view(kernel_depth, kernel_rows, kernel_cols)
    output = torch.empty((output_depth, output_rows, output_cols),
                         device=input.device, dtype=input.dtype)

    stream = torch.cuda.current_stream()

    TILE = 128

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        ROWS_PER_BLOCK = 4
        DEPTHS_PER_BLOCK = 5
        occupancy = 4
        grid = (
            ct.cdiv(output_cols, TILE),
            ct.cdiv(output_rows, ROWS_PER_BLOCK),
            ct.cdiv(output_depth, DEPTHS_PER_BLOCK),
        )
        ct.launch(
            stream,
            grid,
            _conv3d_3x3x3_d5_rows4_kc_outer_kernel,
            (input_3d, kernel_3d, output, ROWS_PER_BLOCK, DEPTHS_PER_BLOCK, TILE),
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": TILE,
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
            "DEPTHS_PER_BLOCK": DEPTHS_PER_BLOCK,
            "occupancy": occupancy,
            "variant": "3x3x3_d5_rows4_kc_outer_tile128",
        })
    else:
        ROWS_PER_BLOCK = 1
        occupancy = 4
        grid = (ct.cdiv(output_cols, TILE), output_rows, output_depth)
        ct.launch(
            stream,
            grid,
            _conv3d_generic_kernel,
            (input_3d, kernel_3d, output,
             kernel_depth, kernel_rows, kernel_cols, TILE),
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "TILE": TILE,
            "ROWS_PER_BLOCK": ROWS_PER_BLOCK,
            "KERNEL_DEPTH": kernel_depth,
            "KERNEL_ROWS": kernel_rows,
            "KERNEL_COLS": kernel_cols,
            "occupancy": occupancy,
            "variant": "generic",
        })

    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
