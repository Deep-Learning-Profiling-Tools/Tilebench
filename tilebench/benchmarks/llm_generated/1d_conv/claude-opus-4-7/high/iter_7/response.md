Looking at the trajectory:
- Best Triton was iter 4 (BLOCK_N=64) at 7.9%; iter 6 (BLOCK_N=256) regressed to 5.2%
- Best cuTile is iter 6 (BLOCK_N=256, BLOCK_K=32) at 9.8% stop_score / 14.2% on the biggest fp16 case

The matmul approach is good (loads only ~1.5× redundant data), but the fp32 path with `tl.static_range(0, 127)` over-unrolls. My iter 7 plan:
1. **Triton fp16**: keep BLOCK_M=64×BLOCK_N=256 tile, but bump BLOCK_K to 64 (→ NUM_K_ITERS=6) for better MMA throughput, and num_stages=3 for deeper pipeline.
2. **Triton fp32**: switch the 127-iter loop from `tl.static_range` to runtime `tl.range(..., num_stages=2)` to keep the kernel small and software-pipeline the loads; bump BLOCK_SIZE=8192.
3. **cuTile fp16**: same BLOCK_K=64 bump for higher MMA throughput.
4. **cuTile fp32**: unchanged (best so far).

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_matmul_kernel(x_ptr, w_ptr, out_ptr, input_size, output_size,
                          KERNEL_SIZE: tl.constexpr,
                          BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr,
                          NUM_K_ITERS: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_M * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in tl.range(0, NUM_K_ITERS):
        k_start = kk * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_addrs = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
        x_mask = x_addrs < input_size
        X = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

        k_minus_n = offs_k[:, None] - offs_n[None, :]
        w_valid = (k_minus_n >= 0) & (k_minus_n < KERNEL_SIZE)
        w_idx = tl.where(w_valid, k_minus_n, 0)
        W = tl.load(w_ptr + w_idx, mask=w_valid, other=0.0)

        acc = tl.dot(X, W, acc)

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
    # Runtime loop (NOT unrolled) — avoids 127-way unrolling that bloats the kernel
    # and prevents the LD pipeline from being effective.
    for j in tl.range(0, KERNEL_SIZE, num_stages=2):
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
        # MMA path — fp16/bf16 use Tensor Cores with fp32 accumulator
        BLOCK_M = 64
        BLOCK_N = 256
        BLOCK_K = 64                       # bumped from 32 → 64 for higher per-iter MMA throughput
        # ceil((BLOCK_N + KERNEL_SIZE - 1) / BLOCK_K)
        NUM_K_ITERS = (BLOCK_N + kernel_size - 1 + BLOCK_K - 1) // BLOCK_K  # = 6 for K=127
        num_warps = 8
        num_stages = 3

        grid = (triton.cdiv(output_size, BLOCK_M * BLOCK_N),)
        _conv1d_matmul_kernel[grid](
            input, kernel, output, input_size, output_size,
            KERNEL_SIZE=kernel_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            NUM_K_ITERS=NUM_K_ITERS,
            num_warps=num_warps, num_stages=num_stages,
        )
        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "matmul",
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
            "NUM_K_ITERS": NUM_K_ITERS,
            "num_warps": num_warps, "num_stages": num_stages,
        })
    else:
        # fp32 path — explicit fp32 scalar (TF32 marginal for atol=0.1)
        BLOCK_SIZE = 8192
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
                          BLOCK_N: ConstInt,
                          BLOCK_K: ConstInt,
                          NUM_K_ITERS: ConstInt,
                          KERNEL_SIZE: ConstInt):
    bid = ct.bid(0)
    base = bid * BLOCK_M * BLOCK_N

    offs_m = ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = ct.arange(BLOCK_N, dtype=np.int32)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)

    for kk in range(NUM_K_ITERS):
        offs_k = ct.arange(BLOCK_K, dtype=np.int32) + (kk * BLOCK_K)

        x_idx = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
        X = ct.gather(x, x_idx, padding_value=0.0)

        k_minus_n = offs_k[:, None] - offs_n[None, :]
        W = ct.gather(w, k_minus_n, padding_value=0.0)

        acc = ct.mma(X, W, acc)

    out_idx = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    ct.scatter(out, out_idx, ct.astype(acc, out.dtype))


@ct.kernel(occupancy=4)
def _conv1d_scalar_kernel(x, w, out, output_size,
                          KERNEL_SIZE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    base = bid * TILE
    offs = base + ct.arange(TILE, dtype=np.int32)

    acc = ct.full((TILE,), 0.0, dtype=np.float32)
    for j in range(KERNEL_SIZE):
        xj = ct.gather(x, offs + j, padding_value=0.0)
        wj = ct.load(w, index=(j,), shape=())
        acc = acc + ct.astype(xj, np.float32) * ct.astype(wj, np.float32)

    ct.scatter(out, offs, ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    if input.dtype in (torch.float16, torch.bfloat16):
        # MMA path — Tensor Cores with fp32 accumulator
        BLOCK_M = 64
        BLOCK_N = 256
        BLOCK_K = 64                       # bumped 32 → 64 for higher MMA throughput
        # ceil((BLOCK_N + KERNEL_SIZE - 1) / BLOCK_K) = ceil(382/64) = 6
        NUM_K_ITERS = (BLOCK_N + kernel_size - 1 + BLOCK_K - 1) // BLOCK_K
        occupancy = 2

        grid = (ct.cdiv(output_size, BLOCK_M * BLOCK_N), 1, 1)
        ct.launch(stream, grid, _conv1d_matmul_kernel,
                  (input, kernel, output, input_size, output_size,
                   BLOCK_M, BLOCK_N, BLOCK_K, NUM_K_ITERS, kernel_size))

        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "matmul",
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
            "NUM_K_ITERS": NUM_K_ITERS,
            "occupancy": occupancy,
        })
    else:
        TILE = 2048
        occupancy = 4
        grid = (ct.cdiv(output_size, TILE), 1, 1)
        ct.launch(stream, grid, _conv1d_scalar_kernel,
                  (input, kernel, output, output_size, kernel_size, TILE))

        _LAST_CFG.clear()
        _LAST_CFG.update({
            "kernel": "scalar",
            "TILE": TILE,
            "occupancy": occupancy,
        })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```
