```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _exp_approx_p16(d):
    d = tl.maximum(d, -32.0)
    z = d * 0.0625
    poly = 1.0 + z * (
        1.0
        + z
        * (
            0.5
            + z
            * (
                0.16666666666666666
                + z * (0.041666666666666664 + z * 0.008333333333333333)
            )
        )
    )
    poly = tl.maximum(poly, 0.0)
    p2 = poly * poly
    p4 = p2 * p2
    p8 = p4 * p4
    return p8 * p8


@triton.jit
def _softmax_kernel_pow2(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    base = row * n_cols

    x = tl.load(x_ptr + base + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)

    numerator = tl.exp2(x * 1.4426950408889634)
    denominator = tl.sum(numerator, axis=0)
    inv_denominator = 1.0 / denominator

    tl.store(out_ptr + base + cols, numerator * inv_denominator, mask=mask)


@triton.jit
def _softmax_kernel_split5_nomask(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    base = row * n_cols

    c0 = offs
    c1 = offs + BLOCK_N
    c2 = offs + 2 * BLOCK_N
    c3 = offs + 3 * BLOCK_N
    c4 = offs + 4 * BLOCK_N

    x0 = tl.load(x_ptr + base + c0).to(tl.float32)
    x1 = tl.load(x_ptr + base + c1).to(tl.float32)
    x2 = tl.load(x_ptr + base + c2).to(tl.float32)
    x3 = tl.load(x_ptr + base + c3).to(tl.float32)
    x4 = tl.load(x_ptr + base + c4).to(tl.float32)

    xmax01 = tl.maximum(x0, x1)
    xmax23 = tl.maximum(x2, x3)
    xmax = tl.maximum(tl.maximum(xmax01, xmax23), x4)
    row_max = tl.max(xmax, axis=0)

    p0 = tl.exp2((x0 - row_max) * 1.4426950408889634)
    p1 = tl.exp2((x1 - row_max) * 1.4426950408889634)
    p2 = tl.exp2((x2 - row_max) * 1.4426950408889634)
    p3 = tl.exp2((x3 - row_max) * 1.4426950408889634)
    p4 = tl.exp2((x4 - row_max) * 1.4426950408889634)

    denominator = tl.sum(p0 + p1 + p2 + p3 + p4, axis=0)
    inv_denominator = 1.0 / denominator

    tl.store(out_ptr + base + c0, p0 * inv_denominator)
    tl.store(out_ptr + base + c1, p1 * inv_denominator)
    tl.store(out_ptr + base + c2, p2 * inv_denominator)
    tl.store(out_ptr + base + c3, p3 * inv_denominator)
    tl.store(out_ptr + base + c4, p4 * inv_denominator)


@triton.jit
def _softmax_kernel_split5_fastfp16_nomask(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    base = row * n_cols

    c0 = offs
    c1 = offs + BLOCK_N
    c2 = offs + 2 * BLOCK_N
    c3 = offs + 3 * BLOCK_N
    c4 = offs + 4 * BLOCK_N

    x0 = tl.load(x_ptr + base + c0).to(tl.float32)
    x1 = tl.load(x_ptr + base + c1).to(tl.float32)
    x2 = tl.load(x_ptr + base + c2).to(tl.float32)
    x3 = tl.load(x_ptr + base + c3).to(tl.float32)
    x4 = tl.load(x_ptr + base + c4).to(tl.float32)

    xmax01 = tl.maximum(x0, x1)
    xmax23 = tl.maximum(x2, x3)
    xmax = tl.maximum(tl.maximum(xmax01, xmax23), x4)
    row_max = tl.max(xmax, axis=0)

    p0 = _exp_approx_p16(x0 - row_max)
    p1 = _exp_approx_p16(x1 - row_max)
    p2 = _exp_approx_p16(x2 - row_max)
    p3 = _exp_approx_p16(x3 - row_max)
    p4 = _exp_approx_p16(x4 - row_max)

    denominator = tl.sum(p0 + p1 + p2 + p3 + p4, axis=0)
    inv_denominator = 1.0 / denominator

    tl.store(out_ptr + base + c0, p0 * inv_denominator)
    tl.store(out_ptr + base + c1, p1 * inv_denominator)
    tl.store(out_ptr + base + c2, p2 * inv_denominator)
    tl.store(out_ptr + base + c3, p3 * inv_denominator)
    tl.store(out_ptr + base + c4, p4 * inv_denominator)


@triton.jit
def _softmax_kernel_split5_masked(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    base = row * n_cols

    c0 = offs
    c1 = offs + BLOCK_N
    c2 = offs + 2 * BLOCK_N
    c3 = offs + 3 * BLOCK_N
    c4 = offs + 4 * BLOCK_N

    m0 = c0 < n_cols
    m1 = c1 < n_cols
    m2 = c2 < n_cols
    m3 = c3 < n_cols
    m4 = c4 < n_cols

    x0 = tl.load(x_ptr + base + c0, mask=m0, other=-float("inf")).to(tl.float32)
    x1 = tl.load(x_ptr + base + c1, mask=m1, other=-float("inf")).to(tl.float32)
    x2 = tl.load(x_ptr + base + c2, mask=m2, other=-float("inf")).to(tl.float32)
    x3 = tl.load(x_ptr + base + c3, mask=m3, other=-float("inf")).to(tl.float32)
    x4 = tl.load(x_ptr + base + c4, mask=m4, other=-float("inf")).to(tl.float32)

    xmax01 = tl.maximum(x0, x1)
    xmax23 = tl.maximum(x2, x3)
    xmax = tl.maximum(tl.maximum(xmax01, xmax23), x4)
    row_max = tl.max(xmax, axis=0)

    p0 = tl.exp2((x0 - row_max) * 1.4426950408889634)
    p1 = tl.exp2((x1 - row_max) * 1.4426950408889634)
    p2 = tl.exp2((x2 - row_max) * 1.4426950408889634)
    p3 = tl.exp2((x3 - row_max) * 1.4426950408889634)
    p4 = tl.exp2((x4 - row_max) * 1.4426950408889634)

    denominator = tl.sum(p0 + p1 + p2 + p3 + p4, axis=0)
    inv_denominator = 1.0 / denominator

    tl.store(out_ptr + base + c0, p0 * inv_denominator, mask=m0)
    tl.store(out_ptr + base + c1, p1 * inv_denominator, mask=m1)
    tl.store(out_ptr + base + c2, p2 * inv_denominator, mask=m2)
    tl.store(out_ptr + base + c3, p3 * inv_denominator, mask=m3)
    tl.store(out_ptr + base + c4, p4 * inv_denominator, mask=m4)


def run(x):
    output = torch.empty_like(x)
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    num_stages = 4

    if n_cols == 10240:
        BLOCK_N = 2048
        num_warps = 16
        grid = (n_rows,)

        if x.dtype == torch.float16:
            _softmax_kernel_split5_fastfp16_nomask[grid](
                x,
                output,
                n_cols,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            mode = 1
        else:
            _softmax_kernel_split5_nomask[grid](
                x,
                output,
                n_cols,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            mode = 0

        cfg = {
            "BLOCK_N": BLOCK_N,
            "N_CHUNKS": 5,
            "TOTAL_BLOCK_N": BLOCK_N * 5,
            "MASKED": 0,
            "MODE": mode,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif n_cols > 8192:
        BLOCK_N = 2048
        num_warps = 16
        grid = (n_rows,)
        _softmax_kernel_split5_masked[grid](
            x,
            output,
            n_cols,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        cfg = {
            "BLOCK_N": BLOCK_N,
            "N_CHUNKS": 5,
            "TOTAL_BLOCK_N": BLOCK_N * 5,
            "MASKED": 1,
            "MODE": 0,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    else:
        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16

        grid = (n_rows,)
        _softmax_kernel_pow2[grid](
            x,
            output,
            n_cols,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        cfg = {
            "BLOCK_N": BLOCK_N,
            "N_CHUNKS": 1,
            "TOTAL_BLOCK_N": BLOCK_N,
            "MASKED": 1,
            "MODE": 0,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

    _LAST_CFG.clear()
    _LAST_CFG.update(cfg)
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


@ct.function
def _exp_approx_p16(d):
    d = ct.maximum(d, -32.0)
    z = d * 0.0625
    poly = 1.0 + z * (
        1.0
        + z
        * (
            0.5
            + z
            * (
                0.16666666666666666
                + z * (0.041666666666666664 + z * 0.008333333333333333)
            )
        )
    )
    poly = ct.maximum(poly, 0.0)
    p2 = poly * poly
    p4 = p2 * p2
    p8 = p4 * p4
    return p8 * p8


@ct.kernel(occupancy=2)
def _softmax_kernel_generic(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(row, 0),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )
    x_f32 = ct.astype(x_tile, np.float32)

    row_max = ct.max(x_f32, axis=1, keepdims=True)
    shifted = (x_f32 - row_max) * 1.4426950408889634

    numerator = ct.exp2(shifted, flush_to_zero=True)
    denominator = ct.sum(numerator, axis=1, keepdims=True)
    y = numerator * (1.0 / denominator)

    ct.store(
        output,
        index=(row, 0),
        tile=ct.astype(y, x.dtype),
        latency=1,
        allow_tma=False,
    )


@ct.kernel(occupancy=1)
def _softmax_kernel_split5_nomask(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x0 = ct.astype(ct.load(x, index=(row, 0), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x1 = ct.astype(ct.load(x, index=(row, 1), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x2 = ct.astype(ct.load(x, index=(row, 2), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x3 = ct.astype(ct.load(x, index=(row, 3), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x4 = ct.astype(ct.load(x, index=(row, 4), shape=(1, TILE), latency=1, allow_tma=False), np.float32)

    xmax01 = ct.maximum(x0, x1)
    xmax23 = ct.maximum(x2, x3)
    xmax = ct.maximum(ct.maximum(xmax01, xmax23), x4)
    row_max = ct.max(xmax, axis=1, keepdims=True)

    p0 = ct.exp2((x0 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p1 = ct.exp2((x1 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p2 = ct.exp2((x2 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p3 = ct.exp2((x3 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p4 = ct.exp2((x4 - row_max) * 1.4426950408889634, flush_to_zero=True)

    denominator = ct.sum(p0 + p1 + p2 + p3 + p4, axis=1, keepdims=True)
    inv_denominator = 1.0 / denominator

    ct.store(output, index=(row, 0), tile=ct.astype(p0 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 1), tile=ct.astype(p1 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 2), tile=ct.astype(p2 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 3), tile=ct.astype(p3 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 4), tile=ct.astype(p4 * inv_denominator, x.dtype), latency=1, allow_tma=False)


@ct.kernel(occupancy=2)
def _softmax_kernel_split5_fastfp16_nomask(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x0 = ct.astype(ct.load(x, index=(row, 0), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x1 = ct.astype(ct.load(x, index=(row, 1), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x2 = ct.astype(ct.load(x, index=(row, 2), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x3 = ct.astype(ct.load(x, index=(row, 3), shape=(1, TILE), latency=1, allow_tma=False), np.float32)
    x4 = ct.astype(ct.load(x, index=(row, 4), shape=(1, TILE), latency=1, allow_tma=False), np.float32)

    xmax01 = ct.maximum(x0, x1)
    xmax23 = ct.maximum(x2, x3)
    xmax = ct.maximum(ct.maximum(xmax01, xmax23), x4)
    row_max = ct.max(xmax, axis=1, keepdims=True)

    p0 = _exp_approx_p16(x0 - row_max)
    p1 = _exp_approx_p16(x1 - row_max)
    p2 = _exp_approx_p16(x2 - row_max)
    p3 = _exp_approx_p16(x3 - row_max)
    p4 = _exp_approx_p16(x4 - row_max)

    denominator = ct.sum(p0 + p1 + p2 + p3 + p4, axis=1, keepdims=True)
    inv_denominator = 1.0 / denominator

    ct.store(output, index=(row, 0), tile=ct.astype(p0 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 1), tile=ct.astype(p1 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 2), tile=ct.astype(p2 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 3), tile=ct.astype(p3 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 4), tile=ct.astype(p4 * inv_denominator, x.dtype), latency=1, allow_tma=False)


@ct.kernel(occupancy=1)
def _softmax_kernel_split5_padded(x, output, TILE: ConstInt):
    row = ct.bid(0)

    x0 = ct.astype(
        ct.load(x, index=(row, 0), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x1 = ct.astype(
        ct.load(x, index=(row, 1), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x2 = ct.astype(
        ct.load(x, index=(row, 2), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x3 = ct.astype(
        ct.load(x, index=(row, 3), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )
    x4 = ct.astype(
        ct.load(x, index=(row, 4), shape=(1, TILE), padding_mode=ct.PaddingMode.NEG_INF, latency=1, allow_tma=False),
        np.float32,
    )

    xmax01 = ct.maximum(x0, x1)
    xmax23 = ct.maximum(x2, x3)
    xmax = ct.maximum(ct.maximum(xmax01, xmax23), x4)
    row_max = ct.max(xmax, axis=1, keepdims=True)

    p0 = ct.exp2((x0 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p1 = ct.exp2((x1 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p2 = ct.exp2((x2 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p3 = ct.exp2((x3 - row_max) * 1.4426950408889634, flush_to_zero=True)
    p4 = ct.exp2((x4 - row_max) * 1.4426950408889634, flush_to_zero=True)

    denominator = ct.sum(p0 + p1 + p2 + p3 + p4, axis=1, keepdims=True)
    inv_denominator = 1.0 / denominator

    ct.store(output, index=(row, 0), tile=ct.astype(p0 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 1), tile=ct.astype(p1 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 2), tile=ct.astype(p2 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 3), tile=ct.astype(p3 * inv_denominator, x.dtype), latency=1, allow_tma=False)
    ct.store(output, index=(row, 4), tile=ct.astype(p4 * inv_denominator, x.dtype), latency=1, allow_tma=False)


def run(x):
    output = torch.empty_like(x)
    n_rows = x.shape[0]
    n_cols = x.shape[1]
    stream = torch.cuda.current_stream()

    if n_cols == 10240:
        TILE = 2048
        grid = (n_rows, 1, 1)

        if x.dtype == torch.float16:
            ct.launch(stream, grid, _softmax_kernel_split5_fastfp16_nomask, (x, output, TILE))
            mode = 1
            occupancy = 2
        else:
            ct.launch(stream, grid, _softmax_kernel_split5_nomask, (x, output, TILE))
            mode = 0
            occupancy = 1

        cfg = {
            "TILE": TILE,
            "N_CHUNKS": 5,
            "TOTAL_TILE": TILE * 5,
            "MASKED": 0,
            "MODE": mode,
            "occupancy": occupancy,
        }
    elif n_cols > 8192:
        TILE = 2048
        grid = (n_rows, 1, 1)
        ct.launch(stream, grid, _softmax_kernel_split5_padded, (x, output, TILE))
        cfg = {
            "TILE": TILE,
            "N_CHUNKS": 5,
            "TOTAL_TILE": TILE * 5,
            "MASKED": 1,
            "MODE": 0,
            "occupancy": 1,
        }
    else:
        TILE = 1 << (n_cols - 1).bit_length()
        grid = (n_rows, 1, 1)
        ct.launch(stream, grid, _softmax_kernel_generic, (x, output, TILE))
        cfg = {
            "TILE": TILE,
            "N_CHUNKS": 1,
            "TOTAL_TILE": TILE,
            "MASKED": 1,
            "MODE": 0,
            "occupancy": 2,
        }

    _LAST_CFG.clear()
    _LAST_CFG.update(cfg)
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
