import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _build_b_kernel(kernel_ptr, bmat_ptr,
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
    tl.store(bmat_ptr + offs, vals, mask=mask)


@triton.jit
def _conv1d_mma_dense_kernel(input_ptr, bmat_ptr, output_ptr,
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
            eviction_policy="evict_first",
        )

        b_idx = k[:, None] * BLOCK_N + offs_n[None, :]
        b = tl.load(
            bmat_ptr + b_idx,
            eviction_policy="evict_last",
        )

        acc = tl.dot(a, b, acc)

    out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_idx < output_size
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def _conv1d_fp32_cached_direct_kernel(input_ptr, kernel_ptr, output_ptr,
                                      input_size, output_size,
                                      KERNEL_SIZE: tl.constexpr,
                                      DIRECT_BLOCK: tl.constexpr,
                                      IN_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * DIRECT_BLOCK

    offs_in = tl.arange(0, IN_BLOCK)
    in_mask = (base + offs_in) < input_size
    in_tile = tl.load(
        input_ptr + base + offs_in,
        mask=in_mask,
        other=0.0,
        eviction_policy="evict_first",
    )

    offs_o = tl.arange(0, DIRECT_BLOCK)
    out_idx = base + offs_o
    out_mask = out_idx < output_size

    acc = tl.zeros((DIRECT_BLOCK,), dtype=tl.float32)

    for j in tl.static_range(0, KERNEL_SIZE):
        x = tl.gather(in_tile, offs_o + j, 0).to(tl.float32)
        w = tl.load(kernel_ptr + j, eviction_policy="evict_last").to(tl.float32)
        acc += x * w

    tl.store(output_ptr + out_idx, acc, mask=out_mask)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)

    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 64
    K_TOTAL_PAD = triton.cdiv(kernel_size + BLOCK_N - 1, BLOCK_K) * BLOCK_K
    MMA_BLOCK_OUT = BLOCK_M * BLOCK_N

    DIRECT_BLOCK = 1024
    IN_BLOCK = triton.next_power_of_2(DIRECT_BLOCK + kernel_size - 1)

    BUILD_BLOCK = 1024
    build_num_warps = 4
    build_num_stages = 2

    mma_num_warps = 8
    mma_num_stages = 2

    direct_num_warps = 4
    direct_num_stages = 2

    if input.dtype == torch.float32:
        grid = (triton.cdiv(output_size, DIRECT_BLOCK),)
        _conv1d_fp32_cached_direct_kernel[grid](
            input, kernel, output,
            input_size, output_size,
            KERNEL_SIZE=kernel_size,
            DIRECT_BLOCK=DIRECT_BLOCK,
            IN_BLOCK=IN_BLOCK,
            num_warps=direct_num_warps,
            num_stages=direct_num_stages,
        )
    else:
        bmat = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=input.dtype)
        total_b = K_TOTAL_PAD * BLOCK_N

        build_grid = (triton.cdiv(total_b, BUILD_BLOCK),)
        _build_b_kernel[build_grid](
            kernel, bmat,
            kernel_size,
            total_b,
            BLOCK_N=BLOCK_N,
            BUILD_BLOCK=BUILD_BLOCK,
            num_warps=build_num_warps,
            num_stages=build_num_stages,
        )

        grid = (triton.cdiv(output_size, MMA_BLOCK_OUT),)
        _conv1d_mma_dense_kernel[grid](
            input, bmat, output,
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
        "DIRECT_BLOCK": DIRECT_BLOCK,
        "IN_BLOCK": IN_BLOCK,
        "KERNEL_SIZE": kernel_size,
        "BUILD_BLOCK": BUILD_BLOCK,
        "build_num_warps": build_num_warps,
        "build_num_stages": build_num_stages,
        "mma_num_warps": mma_num_warps,
        "mma_num_stages": mma_num_stages,
        "direct_num_warps": direct_num_warps,
        "direct_num_stages": direct_num_stages,
        "FP32_CACHED_DIRECT": True,
        "DENSE_B_NON_FP32": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
