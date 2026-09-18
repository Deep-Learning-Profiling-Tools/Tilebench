import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_int8_packed_interleave_pair_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    Kb,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PACK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ak == 1)
    tl.assume(stride_bn == 1)
    tl.assume(stride_cn == 1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, PACK_K)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, PACK_K), PACK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in tl.range(0, Kb, PACK_K, num_stages=LOOP_STAGES):
        kb_offsets = kb_start + offs_k

        b_ptrs = b_ptr + kb_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_i32 = tl.load(b_ptrs, eviction_policy="evict_last").to(tl.int32)

        a_base = a_ptr + offs_m[:, None] * stride_am + kb_offsets[None, :] * stride_ak

        b0 = ((b_i32 & 3) - 1).to(tl.int8)
        b1 = (((b_i32 >> 2) & 3) - 1).to(tl.int8)
        b01 = tl.trans(tl.interleave(tl.trans(b0), tl.trans(b1)))

        a0 = tl.load(a_base, eviction_policy="evict_first").to(tl.int8)
        a1 = tl.load(a_base + Kb * stride_ak, eviction_policy="evict_first").to(tl.int8)
        a01 = tl.interleave(a0, a1)

        acc = tl.dot(a01, b01, acc, out_dtype=tl.int32)

        b2 = (((b_i32 >> 4) & 3) - 1).to(tl.int8)
        b3 = (((b_i32 >> 6) & 3) - 1).to(tl.int8)
        b23 = tl.trans(tl.interleave(tl.trans(b2), tl.trans(b3)))

        a2 = tl.load(a_base + (2 * Kb) * stride_ak, eviction_policy="evict_first").to(tl.int8)
        a3 = tl.load(a_base + (3 * Kb) * stride_ak, eviction_policy="evict_first").to(tl.int8)
        a23 = tl.interleave(a2, a3)

        acc = tl.dot(a23, b23, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    Kb = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)

    BLOCK_M = 128
    BLOCK_N = 256
    PACK_K = 64
    GROUP_SIZE_M = 8
    LOOP_STAGES = 3
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _matmul_int8_packed_interleave_pair_kernel[grid](
        a,
        b,
        output,
        M,
        N,
        Kb,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        PACK_K=PACK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        LOOP_STAGES=LOOP_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "PACK_K": PACK_K,
            "DOT_K": 2 * PACK_K,
            "GROUP_SIZE_M": GROUP_SIZE_M,
            "LOOP_STAGES": LOOP_STAGES,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "PAIR_FIELDS": 2,
            "INTERLEAVE": 1,
            "LOAD_MASKS": 0,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
