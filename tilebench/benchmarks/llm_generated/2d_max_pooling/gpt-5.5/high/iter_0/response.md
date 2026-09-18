```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool2d_kernel(input_ptr, output_ptr,
                       H: tl.constexpr, W: tl.constexpr,
                       H_OUT: tl.constexpr, W_OUT: tl.constexpr,
                       KERNEL_SIZE: tl.constexpr,
                       STRIDE: tl.constexpr,
                       PADDING: tl.constexpr,
                       BLOCK_W: tl.constexpr):
    pid_w = tl.program_id(0)
    row = tl.program_id(1)

    cols = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    oh = row % H_OUT
    nc = row // H_OUT

    ih_base = oh * STRIDE - PADDING
    iw_base = cols * STRIDE - PADDING

    acc = tl.full((BLOCK_W,), -float("inf"), dtype=tl.float32)

    for kh in tl.static_range(0, KERNEL_SIZE):
        ih = ih_base + kh
        valid_h = (ih >= 0) & (ih < H)

        for kw in tl.static_range(0, KERNEL_SIZE):
            iw = iw_base + kw
            mask = (cols < W_OUT) & valid_h & (iw >= 0) & (iw < W)
            in_offsets = nc * H * W + ih * W + iw
            vals = tl.load(input_ptr + in_offsets, mask=mask, other=-float("inf")).to(tl.float32)
            acc = tl.maximum(acc, vals)

    out_offsets = row * W_OUT + cols
    tl.store(output_ptr + out_offsets, acc, mask=cols < W_OUT)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)

    BLOCK_W = 512
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(W_out, BLOCK_W), rows)
    _max_pool2d_kernel[grid](
        input, output,
        H=H, W=W,
        H_OUT=H_out, W_OUT=W_out,
        KERNEL_SIZE=kernel_size,
        STRIDE=stride,
        PADDING=padding,
        BLOCK_W=BLOCK_W,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_W": BLOCK_W,
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
def _max_pool2d_kernel(input, output,
                       H: ConstInt, W: ConstInt,
                       H_OUT: ConstInt, W_OUT: ConstInt,
                       KERNEL_SIZE: ConstInt,
                       STRIDE: ConstInt,
                       PADDING: ConstInt,
                       TILE: ConstInt):
    block_w = ct.bid(0)
    row = ct.bid(1)

    cols = block_w * TILE + ct.arange(TILE, dtype=np.int32)
    oh = row % H_OUT
    nc = row // H_OUT

    ih_base = oh * STRIDE - PADDING
    iw_base = cols * STRIDE - PADDING

    acc = ct.full((TILE,), -np.inf, dtype=np.float32)

    for kh in range(0, KERNEL_SIZE):
        ih = ih_base + kh
        valid_h = (ih >= 0) & (ih < H)

        for kw in range(0, KERNEL_SIZE):
            iw = iw_base + kw
            valid = (cols < W_OUT) & valid_h & (iw >= 0) & (iw < W)
            idx = nc * H * W + ih * W + iw
            safe_idx = ct.where(valid, idx, -1)
            vals = ct.astype(ct.gather(input, safe_idx, padding_value=-np.inf), np.float32)
            acc = ct.maximum(acc, vals)

    out_tile = ct.reshape(ct.astype(acc, input.dtype), (1, TILE))
    ct.store(output, index=(row, block_w), tile=out_tile, allow_tma=False)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    rows = N * C * H_out

    output = torch.empty((rows, W_out), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    TILE = 512
    occupancy = 4

    grid = (ct.cdiv(W_out, TILE), rows, 1)
    kernel = _max_pool2d_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
        (input, output, H, W, H_out, W_out, kernel_size, stride, padding, TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output.reshape(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
