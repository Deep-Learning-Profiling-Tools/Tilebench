Looking at the trajectory:
- Triton iter 2 broke fp32 verify due to TF32 precision in `tl.dot`, but the matmul approach was great for fp16 (9% / 39×).
- cuTile iter 2 matmul verified but was awful for fp32 (0.1%).

**Strategy**: dispatch by dtype.
- **Triton fp16**: keep iter 2 matmul (verified, 9%).
- **Triton fp32**: revert to iter 1 scalar (verified, 1.6%).
- **cuTile fp16**: keep iter 2 matmul (verified, 6%).
- **cuTile fp32**: NEW — load `(2*TILE,)` via two coalesced tile loads + `ct.cat`, then scalar-loop with `ct.extract` to amortize HBM. Avoids the slow fp32 mma path.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_matmul_kernel(x_ptr, w_ptr, out_ptr, input_size, output_size,
                          KERNEL_SIZE: tl.constexpr,
                          BLOCK_M: tl.constexpr,
                          K_PAD: tl.constexpr,
                          BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_M * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, K_PAD)
    offs_n = tl.arange(0, BLOCK_N)

    x_addrs = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
    x_mask = x_addrs < input_size
    X = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

    k_minus_n = offs_k[:, None] - offs_n[None, :]
    w_valid = (k_minus_n >= 0) & (k_minus_n < KERNEL_SIZE)
    w_idx = tl.where(w_valid, k_minus_n, 0)
    W = tl.load(w_ptr + w_idx, mask=w_valid, other=0.0)

    acc = tl.dot(X, W, out_dtype=tl.float32)

    out_addrs = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_addrs < output_size
    tl.store(out_ptr + out_addrs,
             acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


@triton.jit
def _conv1d_scalar_kernel(x_ptr, w_ptr, out_ptr, input_size, output_size,
                          KERNEL_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offs < output_size

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for j in tl.static_range(0, KERNEL_SIZE):
        x = tl.load(x_ptr + out_offs + j, mask=out_mask, other=0.0).to(tl.float32)
        wj = tl.load(w_ptr + j).to(tl.float32)
        acc = acc + x * wj

    tl.store(out_ptr + out_offs,
             acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    if input.dtype in (torch.float16, torch.bfloat16):
        BLOCK_M = 64
        K_PAD = 256
        BLOCK_N = 16
        num_warps = 8
        num_stages = 2
        grid = (triton.cdiv(output_size, BLOCK_M * BLOCK_N),)
        _conv1d_matmul_kernel[grid](
            input, kernel, output, input_size, output_size,
            KERNEL_SIZE=kernel_size,
            BLOCK_M=BLOCK_M, K_PAD=K_PAD, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "matmul",
            "BLOCK_M": BLOCK_M, "K_PAD": K_PAD, "BLOCK_N": BLOCK_N,
            "num_warps": num_warps, "num_stages": num_stages,
        })
    else:
        BLOCK_SIZE = 2048
        num_warps = 8
        num_stages = 2
        grid = (triton.cdiv(output_size, BLOCK_SIZE),)
        _conv1d_scalar_kernel[grid](
            input, kernel, output, input_size, output_size,
            KERNEL_SIZE=kernel_size, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps, num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "scalar",
            "BLOCK_SIZE": BLOCK_SIZE,
            "num_warps": num_warps, "num_stages": num_stages,
        })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _conv1d_matmul_kernel(x, w, out, input_size, output_size,
                          BLOCK_M: ConstInt,
                          K_PAD: ConstInt,
                          BLOCK_N: ConstInt):
    bid = ct.bid(0)
    base = bid * BLOCK_M * BLOCK_N

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)
    offs_k = ct.arange(K_PAD, dtype=np.int32)
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)

    x_indices = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
    X = ct.gather(x, x_indices, padding_value=0.0)

    k_minus_n = offs_k[:, None] - offs_n[None, :]
    W = ct.gather(w, k_minus_n, padding_value=0.0)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    acc = ct.mma(X, W, acc)

    out_indices = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    ct.scatter(out, out_indices, ct.astype(acc, out.dtype))


@ct.kernel(occupancy=4)
def _conv1d_scalar_kernel(x, w, out, input_size, output_size,
                          KERNEL_SIZE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    t1 = ct.load(x, index=(bid,), shape=(TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    t2 = ct.load(x, index=(bid + 1,), shape=(TILE,),
                 padding_mode=ct.PaddingMode.ZERO)
    big = ct.cat((t1, t2), axis=0)
    big_f32 = ct.astype(big, np.float32)

    acc = ct.full((TILE,), 0.0, dtype=np.float32)
    for j in range(KERNEL_SIZE):
        sub = ct.extract(big_f32, (j,), shape=(TILE,))
        wj = ct.load(w, index=(j,), shape=())
        acc = acc + sub * ct.astype(wj, np.float32)

    ct.store(out, index=(bid,), tile=ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if input.dtype in (torch.float16, torch.bfloat16):
        BLOCK_M = 64
        K_PAD = 256
        BLOCK_N = 16
        grid = (ct.cdiv(output_size, BLOCK_M * BLOCK_N), 1, 1)
        ct.launch(stream, grid, _conv1d_matmul_kernel,
                  (input, kernel, output, input_size, output_size,
                   BLOCK_M, K_PAD, BLOCK_N))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "matmul",
            "BLOCK_M": BLOCK_M, "K_PAD": K_PAD, "BLOCK_N": BLOCK_N,
            "occupancy": 2,
        })
    else:
        TILE = 2048
        grid = (ct.cdiv(output_size, TILE), 1, 1)
        ct.launch(stream, grid, _conv1d_scalar_kernel,
                  (input, kernel, output, input_size, output_size,
                   kernel_size, TILE))
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "scalar",
            "TILE": TILE,
            "occupancy": 4,
        })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```
