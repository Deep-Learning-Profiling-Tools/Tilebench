```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _jacobi_interior_kernel(input, output,
                            ROWS: ConstInt, COLS: ConstInt,
                            BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    top_view = input.slice(0, 0, ROWS - 2).slice(1, 1, COLS - 1)
    bottom_view = input.slice(0, 2, ROWS).slice(1, 1, COLS - 1)
    left_view = input.slice(0, 1, ROWS - 1).slice(1, 0, COLS - 2)
    right_view = input.slice(0, 1, ROWS - 1).slice(1, 2, COLS)
    out_view = output.slice(0, 1, ROWS - 1).slice(1, 1, COLS - 1)

    acc = ct.astype(
        ct.load(
            top_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )
    acc = acc + ct.astype(
        ct.load(
            bottom_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )
    acc = acc + ct.astype(
        ct.load(
            left_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )
    acc = acc + ct.astype(
        ct.load(
            right_view,
            index=(bid_m, bid_n),
            shape=(BLOCK_M, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        ),
        np.float32,
    )

    ct.store(
        out_view,
        index=(bid_m, bid_n),
        tile=ct.astype(acc * 0.25, input.dtype),
        latency=1,
        allow_tma=False,
    )


@ct.kernel(occupancy=4)
def _copy_rows_kernel(input, output,
                      ROWS: ConstInt, COLS: ConstInt,
                      BLOCK_N: ConstInt):
    bid_n = ct.bid(0)

    top = ct.load(
        input,
        index=(0, bid_n),
        shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        output,
        index=(0, bid_n),
        tile=top,
        latency=1,
        allow_tma=False,
    )

    bottom_in = input.slice(0, ROWS - 1, ROWS)
    bottom_out = output.slice(0, ROWS - 1, ROWS)
    bottom = ct.load(
        bottom_in,
        index=(0, bid_n),
        shape=(1, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        bottom_out,
        index=(0, bid_n),
        tile=bottom,
        latency=1,
        allow_tma=False,
    )


@ct.kernel(occupancy=4)
def _copy_cols_kernel(input, output,
                      ROWS: ConstInt, COLS: ConstInt,
                      BLOCK_M: ConstInt):
    bid_m = ct.bid(0)

    left_in = input.slice(0, 1, ROWS - 1).slice(1, 0, 1)
    left_out = output.slice(0, 1, ROWS - 1).slice(1, 0, 1)
    left = ct.load(
        left_in,
        index=(bid_m, 0),
        shape=(BLOCK_M, 1),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        left_out,
        index=(bid_m, 0),
        tile=left,
        latency=1,
        allow_tma=False,
    )

    right_in = input.slice(0, 1, ROWS - 1).slice(1, COLS - 1, COLS)
    right_out = output.slice(0, 1, ROWS - 1).slice(1, COLS - 1, COLS)
    right = ct.load(
        right_in,
        index=(bid_m, 0),
        shape=(BLOCK_M, 1),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        right_out,
        index=(bid_m, 0),
        tile=right,
        latency=1,
        allow_tma=False,
    )


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_M = 8
    BLOCK_N = 1024
    BOUNDARY_N = 1024
    BOUNDARY_M = 256
    occupancy = 2
    boundary_occupancy = 4

    ct.launch(
        stream,
        (ct.cdiv(cols, BOUNDARY_N), 1, 1),
        _copy_rows_kernel,
        (input, output, rows, cols, BOUNDARY_N),
    )

    if rows > 2:
        ct.launch(
            stream,
            (ct.cdiv(rows - 2, BOUNDARY_M), 1, 1),
            _copy_cols_kernel,
            (input, output, rows, cols, BOUNDARY_M),
        )

    if rows > 2 and cols > 2:
        grid = (ct.cdiv(rows - 2, BLOCK_M), ct.cdiv(cols - 2, BLOCK_N), 1)
        ct.launch(
            stream,
            grid,
            _jacobi_interior_kernel,
            (input, output, rows, cols, BLOCK_M, BLOCK_N),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BOUNDARY_N": BOUNDARY_N,
        "BOUNDARY_M": BOUNDARY_M,
        "occupancy": occupancy,
        "boundary_occupancy": boundary_occupancy,
        "sliced_interior": 1,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
