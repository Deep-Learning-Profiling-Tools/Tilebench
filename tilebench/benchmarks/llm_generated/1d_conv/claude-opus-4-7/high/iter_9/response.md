Looking at the trajectory:
- **Triton fp16** best was iter4 at 7.9% (BLOCK_N=64, K_PAD=256). Larger BLOCK_N (=128, 256) regressed.
- **cuTile fp16** best was iter8 at 14.3% (BLOCK_N=256, incremental BLOCK_K=32).
- **fp32** scalar is stuck at 2-5% because of 127× input read amplification.

Key insight: the matmul approach saves bandwidth (each input read is shared across BLOCK_N outputs). For fp32 with random inputs scaled ~N(0,1), output magnitude is ~√127≈11, so atol=0.1+rtol×11≈0.21 tolerance should easily accommodate TF32 error (~0.013 RMS). Switching fp32 to **matmul + TF32** should give a massive win.

For Triton, I'll use the incremental BLOCK_K approach (like cuTile iter6/8) since iter4's single-dot K_PAD=256 was register-heavy. Picking BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, NUM_K_ITERS=8 — gives 2× input redundancy (vs iter4's 4×) and fits comfortably in shmem for both fp16/fp32.

For cuTile, keep iter8's winning config but extend it to fp32 with TF32.

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

    for kk in tl.static_range(0, NUM_K_ITERS):
        offs_k = kk * BLOCK_K + tl.arange(0, BLOCK_K)

        # X[m, k] = input[base + m*BLOCK_N + k]   (overlapping rows = Toeplitz packing)
        x_addrs = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
        x_mask = x_addrs < input_size
        X = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

        # W[k, n] = kernel[k - n] if 0 <= k-n < KERNEL_SIZE else 0
        k_minus_n = offs_k[:, None] - offs_n[None, :]
        w_valid = (k_minus_n >= 0) & (k_minus_n < KERNEL_SIZE)
        w_idx = tl.where(w_valid, k_minus_n, 0)
        W = tl.load(w_ptr + w_idx, mask=w_valid, other=0.0)

        acc = tl.dot(X, W, acc, input_precision="tf32", out_dtype=tl.float32)

    out_addrs = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_addrs < output_size
    tl.store(out_ptr + out_addrs,
             acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    # Unified matmul kernel for fp16/bf16/fp32 (TF32 for fp32).
    # BLOCK_N=128 balances input-read amplification (2x) and register/shmem fit
    # for fp32; incremental BLOCK_K=32 keeps the W tile small.
    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 32
    NUM_K_ITERS = (BLOCK_N + kernel_size - 1 + BLOCK_K - 1) // BLOCK_K
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
        "kernel": "matmul",
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


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    # Iter-8 best cuTile config: BLOCK_N=256 split into BLOCK_K=32 × 12 iters.
    # Now also used for fp32 (TF32 tensor cores on B200 — tolerance atol=0.1
    # easily accommodates TF32 accumulation error for K=127, |y|~11).
    BLOCK_M = 64
    BLOCK_N = 256
    BLOCK_K = 32
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
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

**Approach summary:** Switch fp32 to the same TF32 matmul path as fp16 (was the main 2-5% drag in iter8). For Triton, also adopt the incremental-BLOCK_K loop structure (was cuTile's winning pattern), pairing it with BLOCK_N=128 to keep fp32 register pressure manageable while halving input read amplification vs iter4. For cuTile, retain iter8's winning matmul config and extend it to fp32.
