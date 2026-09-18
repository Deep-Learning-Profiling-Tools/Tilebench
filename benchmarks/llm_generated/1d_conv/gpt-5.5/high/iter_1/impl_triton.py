import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_kernel(input_ptr, kernel_ptr, output_ptr,
                   input_size, output_size, kernel_size,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr, K_TOTAL_PAD: tl.constexpr,
                   DIRECT_BLOCK: tl.constexpr, LOOP_STAGES: tl.constexpr):
    pid = tl.program_id(0)

    if input_ptr.dtype.element_ty == tl.float32:
        offs = pid * DIRECT_BLOCK + tl.arange(0, DIRECT_BLOCK)
        mask = offs < output_size
        acc = tl.zeros((DIRECT_BLOCK,), dtype=tl.float32)

        for j in tl.range(0, kernel_size, 1, num_stages=LOOP_STAGES):
            x = tl.load(input_ptr + offs + j, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(kernel_ptr + j).to(tl.float32)
            acc += x * w

        tl.store(output_ptr + offs, acc, mask=mask)
    else:
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

            acc = tl.dot(a, b, acc, input_precision="ieee")

        out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
        out_mask = out_idx < output_size
        tl.store(output_ptr + out_idx, acc, mask=out_mask)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    K_TOTAL_PAD = triton.cdiv(kernel_size + BLOCK_N - 1, BLOCK_K) * BLOCK_K
    DIRECT_BLOCK = BLOCK_M * BLOCK_N
    LOOP_STAGES = 4
    num_warps = 4
    num_stages = 3

    grid = (triton.cdiv(output_size, DIRECT_BLOCK),)

    _conv1d_kernel[grid](
        input, kernel, output,
        input_size, output_size, kernel_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K_TOTAL_PAD=K_TOTAL_PAD,
        DIRECT_BLOCK=DIRECT_BLOCK,
        LOOP_STAGES=LOOP_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "DIRECT_BLOCK": DIRECT_BLOCK,
        "LOOP_STAGES": LOOP_STAGES,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
