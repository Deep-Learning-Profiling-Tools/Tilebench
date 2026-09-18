import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_kernel(input_ptr, kernel_ptr, output_ptr,
                   input_size, output_size, kernel_size,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, K_TOTAL_PAD: tl.constexpr):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_row_groups = tl.cdiv(output_size, BLOCK_N)
    row_valid = offs_m < num_row_groups

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_TOTAL_PAD, BLOCK_K):
        k = k0 + offs_k

        a_idx = offs_m[:, None] * BLOCK_N + k[None, :]
        a_mask = row_valid[:, None] & (a_idx < input_size)
        a = tl.load(input_ptr + a_idx, mask=a_mask, other=0.0)

        b_idx = k[:, None] - offs_n[None, :]
        b_mask = (b_idx >= 0) & (b_idx < kernel_size)
        b_safe = tl.where(b_mask, b_idx, 0)
        b = tl.load(kernel_ptr + b_safe, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, input_precision="tf32")

    out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_idx < output_size
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    K_TOTAL_PAD = triton.cdiv(kernel_size + BLOCK_N - 1, BLOCK_K) * BLOCK_K
    num_warps = 8
    num_stages = 2

    num_row_groups = triton.cdiv(output_size, BLOCK_N)
    grid = (triton.cdiv(num_row_groups, BLOCK_M),)

    _conv1d_kernel[grid](
        input, kernel, output,
        input_size, output_size, kernel_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K_TOTAL_PAD=K_TOTAL_PAD,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
