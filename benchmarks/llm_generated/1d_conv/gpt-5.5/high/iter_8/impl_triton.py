import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _build_b_half_kernel(kernel_ptr, b_hi_ptr,
                         kernel_size,
                         total_elems,
                         BLOCK_N: tl.constexpr,
                         BUILD_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BUILD_BLOCK + tl.arange(0, BUILD_BLOCK)
    mask = offs < total_elems

    k = offs // BLOCK_N
    n = offs - k * BLOCK_N
    j = k - n

    valid = mask & (j >= 0) & (j < kernel_size)
    j_safe = tl.where(valid, j, 0)

    vals = tl.load(kernel_ptr + j_safe, mask=valid, other=0.0)
    tl.store(b_hi_ptr + offs, vals.to(tl.float16), mask=mask)


@triton.jit
def _build_b_split_kernel(kernel_ptr, b_hi_ptr, b_lo_ptr,
                          kernel_size,
                          total_elems,
                          BLOCK_N: tl.constexpr,
                          BUILD_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BUILD_BLOCK + tl.arange(0, BUILD_BLOCK)
    mask = offs < total_elems

    k = offs // BLOCK_N
    n = offs - k * BLOCK_N
    j = k - n

    valid = mask & (j >= 0) & (j < kernel_size)
    j_safe = tl.where(valid, j, 0)

    vals_f = tl.load(kernel_ptr + j_safe, mask=valid, other=0.0).to(tl.float32)
    vals_hi = vals_f.to(tl.float16)
    vals_lo = (vals_f - vals_hi.to(tl.float32)).to(tl.float16)

    tl.store(b_hi_ptr + offs, vals_hi, mask=mask)
    tl.store(b_lo_ptr + offs, vals_lo, mask=mask)


@triton.jit
def _conv1d_mma_dense_half_kernel(input_ptr, b_hi_ptr, output_ptr,
                                  input_size, output_size,
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
        a = tl.load(
            input_ptr + a_idx,
            mask=a_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float16)

        b_idx = k[:, None] * BLOCK_N + offs_n[None, :]
        b = tl.load(
            b_hi_ptr + b_idx,
            eviction_policy="evict_last",
        )

        acc = tl.dot(a, b, acc)

    out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_idx < output_size
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def _conv1d_mma_dense_split4_kernel(input_ptr, b_hi_ptr, b_lo_ptr, output_ptr,
                                    input_size, output_size,
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
        a_f = tl.load(
            input_ptr + a_idx,
            mask=a_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        a_hi = a_f.to(tl.float16)
        a_lo = (a_f - a_hi.to(tl.float32)).to(tl.float16)

        b_idx = k[:, None] * BLOCK_N + offs_n[None, :]

        b_hi = tl.load(
            b_hi_ptr + b_idx,
            eviction_policy="evict_last",
        )
        acc = tl.dot(a_hi, b_hi, acc)
        acc = tl.dot(a_lo, b_hi, acc)

        b_lo = tl.load(
            b_lo_ptr + b_idx,
            eviction_policy="evict_last",
        )
        acc = tl.dot(a_hi, b_lo, acc)
        acc = tl.dot(a_lo, b_lo, acc)

    out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_idx < output_size
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    K_TOTAL_PAD = triton.cdiv(kernel_size + BLOCK_N - 1, BLOCK_K) * BLOCK_K
    MMA_BLOCK_OUT = BLOCK_M * BLOCK_N

    BUILD_BLOCK = 1024
    build_num_warps = 4
    build_num_stages = 2

    mma_num_warps = 4
    mma_num_stages = 3

    b_hi = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=torch.float16)
    total_b = K_TOTAL_PAD * BLOCK_N
    build_grid = (triton.cdiv(total_b, BUILD_BLOCK),)

    grid = (triton.cdiv(output_size, MMA_BLOCK_OUT),)

    if input.dtype == torch.float32:
        b_lo = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=torch.float16)
        _build_b_split_kernel[build_grid](
            kernel, b_hi, b_lo,
            kernel_size,
            total_b,
            BLOCK_N=BLOCK_N,
            BUILD_BLOCK=BUILD_BLOCK,
            num_warps=build_num_warps,
            num_stages=build_num_stages,
        )
        _conv1d_mma_dense_split4_kernel[grid](
            input, b_hi, b_lo, output,
            input_size, output_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            K_TOTAL_PAD=K_TOTAL_PAD,
            num_warps=mma_num_warps,
            num_stages=mma_num_stages,
        )
    else:
        _build_b_half_kernel[build_grid](
            kernel, b_hi,
            kernel_size,
            total_b,
            BLOCK_N=BLOCK_N,
            BUILD_BLOCK=BUILD_BLOCK,
            num_warps=build_num_warps,
            num_stages=build_num_stages,
        )
        _conv1d_mma_dense_half_kernel[grid](
            input, b_hi, output,
            input_size, output_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            K_TOTAL_PAD=K_TOTAL_PAD,
            num_warps=mma_num_warps,
            num_stages=mma_num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "MMA_BLOCK_OUT": MMA_BLOCK_OUT,
        "KERNEL_SIZE": kernel_size,
        "BUILD_BLOCK": BUILD_BLOCK,
        "build_num_warps": build_num_warps,
        "build_num_stages": build_num_stages,
        "mma_num_warps": mma_num_warps,
        "mma_num_stages": mma_num_stages,
        "BMAT_DTYPE_FP16": True,
        "FP32_FP16_SPLIT4_TC": True,
        "DENSE_B_ALL": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
