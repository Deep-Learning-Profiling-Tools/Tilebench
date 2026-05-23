```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool2d_k3s2_contig_bh8_kernel(input_ptr, output_ptr,
                                       H: tl.constexpr, W: tl.constexpr,
                                       H_OUT: tl.constexpr, W_OUT: tl.constexpr,
                                       PADDING: tl.constexpr,
                                       BLOCK_W: tl.constexpr,
                                       IN_W_BLOCK: tl.constexpr,
                                       NEED_W: tl.constexpr):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    nc = tl.program_id(2)

    local_cols = tl.arange(0, BLOCK_W)
    in_offsets = tl.arange(0, IN_W_BLOCK)

    out_col0 = pid_w * BLOCK_W
    out_cols = out_col0 + local_cols
    mask_out = out_cols < W_OUT

    oh0 = pid_h * 8
    ih_base0 = oh0 * 2 - PADDING
    iw_start = out_col0 * 2 - PADDING

    base_nc = nc * H * W

    idx0 = local_cols * 2
    idx1 = idx0 + 1
    idx2 = idx0 + 2

    acc0 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc1 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc2 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc3 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc4 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc5 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc6 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc7 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)

    in_cols = iw_start + in_offsets
    valid_w = (in_offsets < NEED_W) & (in_cols >= 0) & (in_cols < W)

    for rh in tl.static_range(0, 17):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)

        row_vals = tl.load(
            input_ptr + base_nc + ih * W + in_cols,
            mask=valid_h & valid_w,
            other=-float("inf"),
        ).to(tl.float32)

        v0 = tl.gather(row_vals, idx0, 0)
        v1 = tl.gather(row_vals, idx1, 0)
        v2 = tl.gather(row_vals, idx2, 0)
        hmax = tl.maximum(tl.maximum(v0, v1), v2)

        if rh < 3:
            acc0 = tl.maximum(acc0, hmax)
        if (rh >= 2) and (rh < 5):
            acc1 = tl.maximum(acc1, hmax)
        if (rh >= 4) and (rh < 7):
            acc2 = tl.maximum(acc2, hmax)
        if (rh >= 6) and (rh < 9):
            acc3 = tl.maximum(acc3, hmax)
        if (rh >= 8) and (rh < 11):
            acc4 = tl.maximum(acc4, hmax)
        if (rh >= 10) and (rh < 13):
            acc5 = tl.maximum(acc5, hmax)
        if (rh >= 12) and (rh < 15):
            acc6 = tl.maximum(acc6, hmax)
        if rh >= 14:
            acc7 = tl.maximum(acc7, hmax)

    out_base = nc * H_OUT
    tl.store(output_ptr + (out_base + oh0 + 0) * W_OUT + out_cols,
             acc0, mask=mask_out & ((oh0 + 0) < H_OUT))
    tl.store(output_ptr + (out_base + oh0 + 1) * W_OUT + out_cols,
             acc1, mask=mask_out & ((oh0 + 1) < H_OUT))
    tl.store(output_ptr + (out_base + oh0 + 2) * W_OUT + out_cols,
             acc2, mask=mask_out & ((oh0 + 2) < H_OUT))
    tl.store(output_ptr + (out_base + oh0 + 3) * W_OUT + out_cols,
             acc3, mask=mask_out & ((oh0 + 3) < H_OUT))
    tl.store(output_ptr + (out_base + oh0 + 4) * W_OUT + out_cols,
             acc4, mask=mask_out & ((oh0 + 4) < H_OUT))
    tl.store(output_ptr + (out_base + oh0 + 5) * W_OUT + out_cols,
             acc5, mask=mask_out & ((oh0 + 5) < H_OUT))
    tl.store(output_ptr + (out_base + oh0 + 6) * W_OUT + out_cols,
             acc6, mask=mask_out & ((oh0 + 6) < H_OUT))
    tl.store(output_ptr + (out_base + oh0 + 7) * W_OUT + out_cols,
             acc7, mask=mask_out & ((oh0 + 7) < H_OUT))


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)

    BLOCK_W = 128
    BLOCK_H = 8
    IN_W_BLOCK = 512
    NEED_W = 2 * BLOCK_W + 1
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(W_out, BLOCK_W), triton.cdiv(H_out, BLOCK_H), N * C)
    _max_pool2d_k3s2_contig_bh8_kernel[grid](
        input, output,
        H=H,
        W=W,
        H_OUT=H_out,
        W_OUT=W_out,
        PADDING=padding,
        BLOCK_W=BLOCK_W,
        IN_W_BLOCK=IN_W_BLOCK,
        NEED_W=NEED_W,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_contig_bh8",
        "BLOCK_W": BLOCK_W,
        "BLOCK_H": BLOCK_H,
        "IN_W_BLOCK": IN_W_BLOCK,
        "NEED_W": NEED_W,
        "num_warps": num_warps,
        "num_stages": num_stages,
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


@ct.kernel
def _max_pool2d_k3s2_contig_bh8_kernel(input, output,
                                       H: ConstInt, W: ConstInt,
                                       H_OUT: ConstInt, W_OUT: ConstInt,
                                       PADDING: ConstInt,
                                       TILE: ConstInt,
                                       IN_TILE: ConstInt,
                                       HALF_IN: ConstInt,
                                       NEED_W: ConstInt):
    block_w = ct.bid(0)
    h_block = ct.bid(1)
    nc = ct.bid(2)

    offs = ct.arange(IN_TILE, dtype=np.int32)
    out_col0 = block_w * TILE
    iw_start = out_col0 * 2 - PADDING
    ih_base0 = h_block * 16 - PADDING
    base_nc = nc * H * W

    acc0 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc1 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc2 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc3 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc4 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc5 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc6 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc7 = ct.full((TILE,), -np.inf, dtype=np.float32)

    in_cols = iw_start + offs
    valid_w = (offs < NEED_W) & (in_cols >= 0) & (in_cols < W)

    for rh in range(0, 17):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)
        idx = base_nc + ih * W + in_cols
        safe_idx = ct.where(valid_h & valid_w, idx, -1)

        vals = ct.astype(
            ct.gather(input, safe_idx, padding_value=-np.inf, latency=1),
            np.float32,
        )
        vals2 = ct.reshape(vals, (HALF_IN, 2))

        v0 = ct.reshape(ct.extract(vals2, (0, 0), (TILE, 1)), (TILE,))
        v1 = ct.reshape(ct.extract(vals2, (0, 1), (TILE, 1)), (TILE,))
        v2 = ct.reshape(ct.extract(vals2, (1, 0), (TILE, 1)), (TILE,))
        hmax = ct.maximum(ct.maximum(v0, v1), v2)

        if rh < 3:
            acc0 = ct.maximum(acc0, hmax)
        if (rh >= 2) and (rh < 5):
            acc1 = ct.maximum(acc1, hmax)
        if (rh >= 4) and (rh < 7):
            acc2 = ct.maximum(acc2, hmax)
        if (rh >= 6) and (rh < 9):
            acc3 = ct.maximum(acc3, hmax)
        if (rh >= 8) and (rh < 11):
            acc4 = ct.maximum(acc4, hmax)
        if (rh >= 10) and (rh < 13):
            acc5 = ct.maximum(acc5, hmax)
        if (rh >= 12) and (rh < 15):
            acc6 = ct.maximum(acc6, hmax)
        if rh >= 14:
            acc7 = ct.maximum(acc7, hmax)

    out_row = nc * H_OUT + h_block * 8

    ct.store(output, index=(out_row + 0, block_w),
             tile=ct.reshape(ct.astype(acc0, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 1, block_w),
             tile=ct.reshape(ct.astype(acc1, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 2, block_w),
             tile=ct.reshape(ct.astype(acc2, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 3, block_w),
             tile=ct.reshape(ct.astype(acc3, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 4, block_w),
             tile=ct.reshape(ct.astype(acc4, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 5, block_w),
             tile=ct.reshape(ct.astype(acc5, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 6, block_w),
             tile=ct.reshape(ct.astype(acc6, input.dtype), (1, TILE)),
             allow_tma=False)
    ct.store(output, index=(out_row + 7, block_w),
             tile=ct.reshape(ct.astype(acc7, input.dtype), (1, TILE)),
             allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE = 128
    BLOCK_H = 8
    IN_TILE = 512
    HALF_IN = 256
    NEED_W = 2 * TILE + 1
    occupancy = 4

    grid = (ct.cdiv(W_out, TILE), ct.cdiv(H_out, BLOCK_H), N * C)
    kernel = _max_pool2d_k3s2_contig_bh8_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (input, output, H, W, H_out, W_out, padding, TILE, IN_TILE, HALF_IN, NEED_W),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_contig_bh8",
        "TILE": TILE,
        "BLOCK_H": BLOCK_H,
        "IN_TILE": IN_TILE,
        "HALF_IN": HALF_IN,
        "NEED_W": NEED_W,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
