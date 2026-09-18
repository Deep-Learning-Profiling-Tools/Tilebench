```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv3d_3x3x3_row_mma_kernel(input, kernel, output,
                                 BLOCK_R: ConstInt,
                                 BLOCK_COL: ConstInt,
                                 BLOCK_K: ConstInt):
    bid_c = ct.bid(0)
    bid_r = ct.bid(1)
    od = ct.bid(2)

    row_start = bid_r * BLOCK_R
    col_start = bid_c * BLOCK_COL

    rk = ct.arange(BLOCK_K, dtype=np.int32)[:, None]
    cc = ct.arange(BLOCK_COL, dtype=np.int32)[None, :]
    rr = ct.arange(BLOCK_R, dtype=np.int32)[:, None]
    kk = ct.arange(BLOCK_K, dtype=np.int32)[None, :]
    rel = kk - rr

    rows = row_start + rk
    cols = col_start + cc

    acc = ct.full((BLOCK_R, BLOCK_COL), 0.0, dtype=np.float32)

    for kd in range(0, 3):
        for kc in range(0, 3):
            x = ct.astype(
                ct.gather(
                    input,
                    (od + kd, rows, cols + kc),
                    padding_value=0,
                    check_bounds=True,
                    latency=1,
                ),
                np.float32,
            )

            w0 = ct.astype(
                ct.load(
                    kernel, index=(kd, 0, kc), shape=(1, 1, 1),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False, latency=1,
                ),
                np.float32,
            ).item()
            w1 = ct.astype(
                ct.load(
                    kernel, index=(kd, 1, kc), shape=(1, 1, 1),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False, latency=1,
                ),
                np.float32,
            ).item()
            w2 = ct.astype(
                ct.load(
                    kernel, index=(kd, 2, kc), shape=(1, 1, 1),
                    padding_mode=ct.PaddingMode.ZERO,
                    allow_tma=False, latency=1,
                ),
                np.float32,
            ).item()

            w = ct.full((BLOCK_R, BLOCK_K), 0.0, dtype=np.float32)
            w = ct.where(rel == 0, w0, w)
            w = ct.where(rel == 1, w1, w)
            w = ct.where(rel == 2, w2, w)

            acc = ct.mma(w, x, acc)

    ct.store(
        output,
        index=(od, bid_r, bid_c),
        tile=ct.astype(acc, output.dtype).reshape((1, BLOCK_R, BLOCK_COL)),
        allow_tma=False,
        latency=1,
    )


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

    if kernel_depth == 3 and kernel_rows == 3 and kernel_cols == 3:
        BLOCK_R = 16
        BLOCK_COL = 128
        BLOCK_K = 32
        occupancy = 2

        grid = (
            ct.cdiv(output_cols, BLOCK_COL),
            ct.cdiv(output_rows, BLOCK_R),
            output_depth,
        )
        ct.launch(
            stream,
            grid,
            _conv3d_3x3x3_row_mma_kernel,
            (input_3d, kernel_3d, output, BLOCK_R, BLOCK_COL, BLOCK_K),
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "BLOCK_R": BLOCK_R,
            "BLOCK_COL": BLOCK_COL,
            "BLOCK_K": BLOCK_K,
            "occupancy": occupancy,
            "variant": "3x3x3_row_mma_sparse",
        })
    else:
        TILE = 256
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
            "ROWS_PER_BLOCK": 1,
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
