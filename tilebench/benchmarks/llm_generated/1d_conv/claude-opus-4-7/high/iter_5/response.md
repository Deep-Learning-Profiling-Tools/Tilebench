Looking at the trajectory, the iter-4 matmul approach got 13.7% (fp16) by amortizing work via tensor cores, but had ~3-4× memory amplification because row overlap meant input was re-read many times.

**Key insight**: With `BLOCK_N >= KERNEL_SIZE-1`, consecutive rows of the matmul layout do **not** overlap in input, so the X-tile loads match the unique input range. Using `BLOCK_M=64, BLOCK_N=256, BLOCK_K=32` (K-loop over 12 iters to cover BLOCK_N+K-1=382), memory amplification drops to ~1.23×, leaving room for ~2× the previous fp16 score. Same kernel handles fp32 via TF32 cores.

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

        # X[m, k] = input[base + m*BLOCK_N + k]
        x_addrs = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
        x_mask = x_addrs < input_size
        X = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

        # W[k, n] = kernel[k - n] if 0 <= k-n < KERNEL_SIZE else 0
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


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    # Layout: each CTA owns BLOCK_M rows × BLOCK_N consecutive outputs.
    # With BLOCK_N (=256) > KERNEL_SIZE-1 (=126), rows don't overlap in input
    # → matmul X-tile reads each input element only ~once per CTA.
    BLOCK_M = 64
    BLOCK_N = 256
    BLOCK_K = 32
    total_k = BLOCK_N + kernel_size - 1
    NUM_K_ITERS = (total_k + BLOCK_K - 1) // BLOCK_K   # 12 for K=127
    num_warps = 8
    num_stages = 2

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
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "NUM_K_ITERS": NUM_K_ITERS,
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
def _conv1d_kernel(x, w, out, input_size, output_size,
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

        # X[m, k] = input[base + m*BLOCK_N + k]
        x_idx = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
        X = ct.gather(x, x_idx, padding_value=0.0)

        # W[k, n] = kernel[k - n]   (gather returns padding for negative idx)
        k_minus_n = offs_k[:, None] - offs_n[None, :]
        W = ct.gather(w, k_minus_n, padding_value=0.0)

        acc = ct.mma(X, W, acc)

    out_idx = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    ct.scatter(out, out_idx, ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    # BLOCK_N (=256) > KERNEL_SIZE-1 (=126) → rows non-overlapping in input,
    # minimising memory amplification. 12 K-tiles of 32 cover 382 needed.
    BLOCK_M = 64
    BLOCK_N = 256
    BLOCK_K = 32
    total_k = BLOCK_N + kernel_size - 1
    NUM_K_ITERS = (total_k + BLOCK_K - 1) // BLOCK_K
    occupancy = 2

    grid = (ct.cdiv(output_size, BLOCK_M * BLOCK_N), 1, 1)
    ct.launch(stream, grid, _conv1d_kernel,
              (input, kernel, output, input_size, output_size,
               BLOCK_M, BLOCK_N, BLOCK_K, NUM_K_ITERS, kernel_size))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "NUM_K_ITERS": NUM_K_ITERS,
        "occupancy": occupancy,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

The key change: `BLOCK_N=256` (was 64/32) eliminates per-row input overlap, and a real K-loop with `BLOCK_K=32` keeps shared memory feasible for both fp16 and fp32 while preserving tensor-core use.
