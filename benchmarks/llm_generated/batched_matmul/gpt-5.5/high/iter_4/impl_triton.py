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
    tl.assume(M >= 1)
    tl.assume(N >= 1)
    tl.assume(K >= 1)

    a_base = A_ptr + batch_id * M * K
    b_base = B_ptr + batch_id * K * N
    c_base = C_ptr + batch_id * M * N

    a_block = tl.make_block_ptr(
        base=a_base,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_base,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
        b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        acc = tl.dot(a, b, acc, input_precision="ieee")
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    c_block = tl.make_block_ptr(
        base=c_base,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block, acc, boundary_check=(0, 1))


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    C = torch.empty((BATCH * M * N,), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 2

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
        "use_block_ptr": True,
    })
    return C


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
