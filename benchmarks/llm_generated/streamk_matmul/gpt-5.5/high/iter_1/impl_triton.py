import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _zero_kernel(x_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(x_ptr + offs, z, mask=mask)


@triton.jit
def _cast_kernel(acc_ptr, out_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def _streamk_kernel(a_ptr, b_ptr, acc_ptr,
                    M, N, K,
                    stride_am, stride_ak,
                    stride_bk, stride_bn,
                    BLOCK_M: tl.constexpr,
                    BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr,
                    GROUP_SIZE_M: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    total_iters = num_tiles * iters_per_tile

    iter_start = (pid * total_iters) // num_progs
    iter_end = ((pid + 1) * total_iters) // num_progs

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k_base = tl.arange(0, BLOCK_K)

    while iter_start < iter_end:
        tile_id = iter_start // iters_per_tile
        k_iter_start = iter_start - tile_id * iters_per_tile
        iter_tile_end = tl.minimum(iter_end, (tile_id + 1) * iters_per_tile)
        k_iter_end = iter_tile_end - tile_id * iters_per_tile

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tile_in_group = tile_id - group_id * num_pid_in_group

        pid_m = first_pid_m + (tile_in_group % group_size_m)
        pid_n = tile_in_group // group_size_m

        offs_m = pid_m * BLOCK_M + offs_m_base
        offs_n = pid_n * BLOCK_N + offs_n_base

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for kk in tl.range(k_iter_start, k_iter_end, 1):
            offs_k = kk * BLOCK_K + offs_k_base

            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc = tl.dot(a, b, acc, input_precision="tf32")

        c_offsets = offs_m[:, None] * N + offs_n[None, :]
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

        if (k_iter_start == 0) & (k_iter_end == iters_per_tile):
            tl.store(acc_ptr + c_offsets, acc, mask=c_mask)
        else:
            tl.atomic_add(acc_ptr + c_offsets, acc, sem="relaxed", mask=c_mask)

        iter_start = iter_tile_end


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]
    total = M * N

    output = torch.empty((total,), device=a.device, dtype=a.dtype)
    accum = torch.empty((total,), device=a.device, dtype=torch.float32)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    NUM_SMS = 148
    num_warps = 8
    num_stages = 2

    ZERO_BLOCK_SIZE = 1024
    CAST_BLOCK_SIZE = 1024
    vec_warps = 4
    vec_stages = 2

    _zero_kernel[(triton.cdiv(total, ZERO_BLOCK_SIZE),)](
        accum, total,
        BLOCK_SIZE=ZERO_BLOCK_SIZE,
        num_warps=vec_warps,
        num_stages=vec_stages,
    )

    _streamk_kernel[(NUM_SMS,)](
        a, b, accum,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _cast_kernel[(triton.cdiv(total, CAST_BLOCK_SIZE),)](
        accum, output, total,
        BLOCK_SIZE=CAST_BLOCK_SIZE,
        num_warps=vec_warps,
        num_stages=vec_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "NUM_SMS": NUM_SMS,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "ZERO_BLOCK_SIZE": ZERO_BLOCK_SIZE,
        "CAST_BLOCK_SIZE": CAST_BLOCK_SIZE,
        "vec_warps": vec_warps,
        "vec_stages": vec_stages,
    })
    return output.view(M, N)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
