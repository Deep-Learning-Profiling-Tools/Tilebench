Looking at the trajectory, I'm stuck at ~5.5% roofline. The bottleneck appears to be either L1 cache pressure from many masked loads, or instruction overhead from the mask computation in the inner loop.

For Triton, I'll switch to `tl.make_block_ptr` + `padding_option="zero"`, which can leverage TMA hardware on B200 and avoids manual mask computation — that's typically a big win for stencil-style code on Blackwell.

For cuTile, I'll try a larger block size (BM=64) to amortize the gather overhead and reduce program count.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv2d_kernel(x_ptr, k_ptr, out_ptr,
                   H, W,
                   KH: tl.constexpr, KW: tl.constexpr,
                   PAD_H: tl.constexpr, PAD_W: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    base_m = pid_m * BLOCK_M - PAD_H
    base_n = pid_n * BLOCK_N - PAD_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            x_block = tl.make_block_ptr(
                base=x_ptr,
                shape=(H, W),
                strides=(W, 1),
                offsets=(base_m + kh, base_n + kw),
                block_shape=(BLOCK_M, BLOCK_N),
                order=(1, 0),
            )
            x_val = tl.load(x_block, boundary_check=(0, 1),
                            padding_option="zero").to(tl.float32)
            w_val = tl.load(k_ptr + kh * KW + kw).to(tl.float32)
            acc += x_val * w_val

    out_block = tl.make_block_ptr(
        base=out_ptr,
        shape=(H, W),
        strides=(W, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(out_block, acc.to(out_ptr.dtype.element_ty),
             boundary_check=(0, 1))


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    BLOCK_M = 32
    BLOCK_N = 128
    num_warps = 4
    num_stages = 3

    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(W, BLOCK_N))
    _conv2d_kernel[grid](
        input, kernel, output, H, W,
        KH=KH, KW=KW, PAD_H=PAD_H, PAD_W=PAD_W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
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
def _conv2d_kernel(x, k, out,
                   KH: ConstInt, KW: ConstInt,
                   PAD_H: ConstInt, PAD_W: ConstInt,
                   BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    offs_m = bid_m * BM + ct.arange(BM, dtype=np.int32)   # [BM]
    offs_n = bid_n * BN + ct.arange(BN, dtype=np.int32)   # [BN]

    acc = ct.full((BM, BN), 0.0, dtype=np.float32)

    for kh in range(KH):
        for kw in range(KW):
            row = offs_m[:, None] + (kh - PAD_H)
            col = offs_n[None, :] + (kw - PAD_W)
            row2 = ct.broadcast_to(row, (BM, BN))
            col2 = ct.broadcast_to(col, (BM, BN))
            vals = ct.gather(x, (row2, col2), padding_value=0.0)
            w_val = ct.load(k, index=(kh, kw), shape=())
            acc = acc + ct.astype(vals, np.float32) * ct.astype(w_val, np.float32)

    out_tile = ct.astype(acc, x.dtype)
    ct.store(out, index=(bid_m, bid_n), tile=out_tile)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    x2d = input.view(H, W)
    k2d = kernel.view(KH, KW)
    out2d = output.view(H, W)

    BM = 64
    BN = 128
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(H, BM), ct.cdiv(W, BN), 1)
    ct.launch(stream, grid, _conv2d_kernel,
              (x2d, k2d, out2d, KH, KW, PAD_H, PAD_W, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BM": BM,
        "BN": BN,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

My approach: Triton switches to `make_block_ptr` with `padding_option="zero"` which on B200 can use TMA-style hardware bounds checking instead of manual masking — eliminating ~49× 2D mask arithmetic operations and improving load coalescing. CuTile keeps the gather approach but with BM=64 to halve the program count and reduce per-program overhead.
