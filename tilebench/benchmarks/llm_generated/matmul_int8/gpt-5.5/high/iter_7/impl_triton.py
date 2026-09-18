import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_int8_packed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    Kb,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    tl.assume(Kb > 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    a_row_base = offs_m[:, None] * (Kb * 4)
    b_col_base = offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in tl.range(0, Kb, BLOCK_K, num_stages=LOOP_STAGES):
        kb_offsets = kb_start + offs_k

        b_bytes = tl.load(
            b_ptr + kb_offsets[:, None] * N + b_col_base,
            eviction_policy="evict_last",
        )

        for field in tl.static_range(0, 4):
            a_cols = field * Kb + kb_offsets
            a_tile = tl.load(
                a_ptr + a_row_base + a_cols[None, :],
                eviction_policy="evict_first",
            ).to(tl.int8)

            b_bits_i8 = ((b_bytes >> (2 * field)) & 3).to(tl.int8)
            b_vals = (b_bits_i8 - 1).to(tl.int8)

            acc = tl.dot(a_tile, b_vals, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    Kb = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    LOOP_STAGES = 4
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _matmul_int8_packed_kernel[grid](
        a,
        b,
        output,
        Kb,
        M=M,
        N=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
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
            "BLOCK_K": BLOCK_K,
            "GROUP_SIZE_M": GROUP_SIZE_M,
            "LOOP_STAGES": LOOP_STAGES,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOAD_MASKS": 0,
            "STORE_MASKS": 0,
            "U8_SHIFT_DECODE": 1,
            "CONTIGUOUS_STRIDES": 1,
            "SPECIALIZE_MN": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
