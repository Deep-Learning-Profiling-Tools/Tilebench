import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_int8_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K_b,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KB: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_KB)

    a_base = a_ptr + offs_m[:, None] * stride_am
    b_base = b_ptr + offs_n[None, :] * stride_bn

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in range(0, K_b, BLOCK_KB):
        kb_idx = kb_start + offs_kb
        kb_mask = kb_idx < K_b
        b_packed = tl.load(
            b_base + kb_idx[:, None] * stride_bk,
            mask=kb_mask[:, None] & n_mask,
            other=0,
        )
        for i in tl.static_range(4):
            shifted = (b_packed >> (2 * i)) & 3
            b_val = shifted.to(tl.int8) - 1
            k_offset_a = i * K_b + kb_idx
            a_val = tl.load(
                a_base + k_offset_a[None, :] * stride_ak,
                mask=m_mask & kb_mask[None, :],
                other=0,
            )
            acc = tl.dot(a_val, b_val, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_KB = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_int8_kernel[grid](
        a, b, output,
        M, N, K_b,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_KB=BLOCK_KB,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_KB": BLOCK_KB,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
