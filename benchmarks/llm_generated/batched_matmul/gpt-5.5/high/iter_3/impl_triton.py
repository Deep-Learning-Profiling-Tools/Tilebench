import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _batched_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    BATCH,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = tile_id - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    tl.assume(batch_id >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_batch_base = batch_id * M * K
    b_batch_base = batch_id * K * N
    c_batch_base = batch_id * M * N

    a_ptrs = A_ptr + a_batch_base + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + b_batch_base + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_tile * BLOCK_K + offs_k

        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
            other=0.0,
            eviction_policy="evict_first",
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
            eviction_policy="evict_last",
        )

        acc = tl.dot(a, b, acc, input_precision="ieee")

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = C_ptr + c_batch_base + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (batch_id < BATCH) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    C = torch.empty((BATCH * M * N,), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_warps = 4
    num_stages = 3

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        BATCH,
    )

    _batched_matmul_kernel[grid](
        A,
        B,
        C,
        BATCH,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "input_precision": "ieee",
    })
    return C


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
