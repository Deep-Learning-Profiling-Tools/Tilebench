```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool2d_k3s2_bh4_kernel(input_ptr, output_ptr,
                                H: tl.constexpr, W: tl.constexpr,
                                H_OUT: tl.constexpr, W_OUT: tl.constexpr,
                                PADDING: tl.constexpr,
                                BLOCK_W: tl.constexpr):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    nc = tl.program_id(2)

    cols = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_cols = cols < W_OUT

    oh0 = pid_h * 4
    ih_base0 = oh0 * 2 - PADDING
    iw_base = cols * 2 - PADDING
    base_nc = nc * H * W

    acc0 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc1 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc2 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)
    acc3 = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)

    for rh in tl.static_range(0, 9):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)
        hmax = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)

        for kw in tl.static_range(0, 3):
            iw = iw_base + kw
            mask = mask_cols & valid_h & (iw >= 0) & (iw < W)
            vals = tl.load(input_ptr + base_nc + ih * W + iw,
                           mask=mask, other=-float("inf")).to(tl.float32)
            hmax = tl.maximum(hmax, vals)

        if rh < 3:
            acc0 = tl.maximum(acc0, hmax)
        if (rh >= 2) and (rh < 5):
            acc1 = tl.maximum(acc1, hmax)
        if (rh >= 4) and (rh < 7):
            acc2 = tl.maximum(acc2, hmax)
        if rh >= 6:
            acc3 = tl.maximum(acc3, hmax)

    out_row_base = nc * H_OUT
    oh1 = oh0 + 1
    oh2 = oh0 + 2
    oh3 = oh0 + 3

    tl.store(output_ptr + (out_row_base + oh0) * W_OUT + cols,
             acc0, mask=mask_cols & (oh0 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh1) * W_OUT + cols,
             acc1, mask=mask_cols & (oh1 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh2) * W_OUT + cols,
             acc2, mask=mask_cols & (oh2 < H_OUT))
    tl.store(output_ptr + (out_row_base + oh3) * W_OUT + cols,
             acc3, mask=mask_cols & (oh3 < H_OUT))


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)

    BLOCK_W = 512
    BLOCK_H = 4
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(W_out, BLOCK_W), triton.cdiv(H_out, BLOCK_H), N * C)
    _max_pool2d_k3s2_bh4_kernel[grid](
        input, output,
        H=H,
        W=W,
        H_OUT=H_out,
        W_OUT=W_out,
        PADDING=padding,
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh4",
        "BLOCK_W": BLOCK_W,
        "BLOCK_H": BLOCK_H,
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


@ct.kernel(occupancy=4)
def _max_pool2d_k3s2_bh4_kernel(input, output,
                                H: ConstInt, W: ConstInt,
                                H_OUT: ConstInt, W_OUT: ConstInt,
                                PADDING: ConstInt,
                                TILE: ConstInt):
    block_w = ct.bid(0)
    h_block = ct.bid(1)
    nc = ct.bid(2)

    cols = block_w * TILE + ct.arange(TILE, dtype=np.int32)
    oh0 = h_block * 4
    ih_base0 = oh0 * 2 - PADDING
    iw_base = cols * 2 - PADDING
    base_nc = nc * H * W

    acc0 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc1 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc2 = ct.full((TILE,), -np.inf, dtype=np.float32)
    acc3 = ct.full((TILE,), -np.inf, dtype=np.float32)

    for rh in range(0, 9):
        ih = ih_base0 + rh
        valid_h = (ih >= 0) & (ih < H)
        hmax = ct.full((TILE,), -np.inf, dtype=np.float32)

        for kw in range(0, 3):
            iw = iw_base + kw
            valid = (cols < W_OUT) & valid_h & (iw >= 0) & (iw < W)
            idx = base_nc + ih * W + iw
            safe_idx = ct.where(valid, idx, -1)
            vals = ct.astype(ct.gather(input, safe_idx, padding_value=-np.inf), np.float32)
            hmax = ct.maximum(hmax, vals)

        if rh < 3:
            acc0 = ct.maximum(acc0, hmax)
        if (rh >= 2) and (rh < 5):
            acc1 = ct.maximum(acc1, hmax)
        if (rh >= 4) and (rh < 7):
            acc2 = ct.maximum(acc2, hmax)
        if rh >= 6:
            acc3 = ct.maximum(acc3, hmax)

    out_row = nc * H_OUT + oh0

    out0 = ct.reshape(ct.astype(acc0, input.dtype), (1, TILE))
    out1 = ct.reshape(ct.astype(acc1, input.dtype), (1, TILE))
    out2 = ct.reshape(ct.astype(acc2, input.dtype), (1, TILE))
    out3 = ct.reshape(ct.astype(acc3, input.dtype), (1, TILE))

    ct.store(output, index=(out_row, block_w), tile=out0, allow_tma=False)
    ct.store(output, index=(out_row + 1, block_w), tile=out1, allow_tma=False)
    ct.store(output, index=(out_row + 2, block_w), tile=out2, allow_tma=False)
    ct.store(output, index=(out_row + 3, block_w), tile=out3, allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE = 512
    BLOCK_H = 4
    occupancy = 4

    grid = (ct.cdiv(W_out, TILE), ct.cdiv(H_out, BLOCK_H), N * C)
    ct.launch(
        stream,
        grid,
        _max_pool2d_k3s2_bh4_kernel,
        (input, output, H, W, H_out, W_out, padding, TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "mode": "k3s2_bh4",
        "TILE": TILE,
        "BLOCK_H": BLOCK_H,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
