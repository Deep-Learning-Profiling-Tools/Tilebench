import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_kernel(x_ptr, w_ptr, out_ptr, input_size, output_size,
                   KERNEL_SIZE: tl.constexpr,
                   BLOCK_M: tl.constexpr,
                   K_PAD: tl.constexpr,
                   BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * BLOCK_M * BLOCK_N

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, K_PAD)
    offs_n = tl.arange(0, BLOCK_N)

    # X[m, k] = input[base + m*BLOCK_N + k]
    x_addrs = base + offs_m[:, None] * BLOCK_N + offs_k[None, :]
    x_mask = x_addrs < input_size
    X = tl.load(x_ptr + x_addrs, mask=x_mask, other=0.0)

    # W[k, n] = kernel[k - n] if 0 <= k - n < KERNEL_SIZE else 0
    k_minus_n = offs_k[:, None] - offs_n[None, :]
    w_valid = (k_minus_n >= 0) & (k_minus_n < KERNEL_SIZE)
    w_idx = tl.where(w_valid, k_minus_n, 0)
    W = tl.load(w_ptr + w_idx, mask=w_valid, other=0.0)

    # Tensor-core matmul: acc = X @ W
    acc = tl.dot(X, W, out_dtype=tl.float32)  # [BLOCK_M, BLOCK_N]

    # Store: output[base + m*BN + n] = acc[m, n]
    out_addrs = base + offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_addrs < output_size
    tl.store(out_ptr + out_addrs,
             acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    BLOCK_M = 64
    K_PAD = 256
    BLOCK_N = 16
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(output_size, BLOCK_M * BLOCK_N),)
    _conv1d_kernel[grid](
        input, kernel, output, input_size, output_size,
        KERNEL_SIZE=kernel_size,
        BLOCK_M=BLOCK_M, K_PAD=K_PAD, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "K_PAD": K_PAD, "BLOCK_N": BLOCK_N,
        "num_warps": num_warps, "num_stages": num_stages,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
