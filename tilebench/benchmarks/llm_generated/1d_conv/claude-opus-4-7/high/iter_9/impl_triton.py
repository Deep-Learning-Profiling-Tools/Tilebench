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
