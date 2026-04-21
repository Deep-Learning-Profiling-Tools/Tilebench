import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUPSIZE": 8,
    "num_warps": 4,
    "num_stages": 2,
}


@triton.jit
def _bmm_kernel(a_ptr, b_ptr, c_ptr,
                BATCH, M, N, K,
                BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr,
                BLOCK_SIZE_K: tl.constexpr,
                GROUPSIZE: tl.constexpr):
    hw_pid0 = tl.program_id(0)
    hw_pid1 = tl.program_id(1)
    num_programs_pid0 = tl.num_programs(0)
    num_programs_pid1 = tl.num_programs(1)
    pid0, pid1 = tl.swizzle2d(hw_pid0, hw_pid1, num_programs_pid0, num_programs_pid1, GROUPSIZE)
    pid2 = tl.program_id(2)

    offsets_M = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = pid1 * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    mask_M = offsets_M < M
    mask_N = offsets_N < N
    mask_K = offsets_K < K

    A_offsets = pid2 * M * K + offsets_M[:, None] * K + offsets_K[None, :]
    B_offsets = pid2 * K * N + offsets_K[:, None] * N + offsets_N[None, :]

    A_mask = mask_M[:, None] & mask_K[None, :]
    B_mask = mask_K[:, None] & mask_N[None, :]

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for current_k_index in range(0, K, BLOCK_SIZE_K):
        if current_k_index + BLOCK_SIZE_K >= K:
            current_mask_K = (offsets_K + current_k_index) < K
            A_mask = mask_M[:, None] & current_mask_K[None, :]
            B_mask = current_mask_K[:, None] & mask_N[None, :]

        A_data = tl.load(a_ptr + A_offsets, mask=A_mask, other=0.0)
        B_data = tl.load(b_ptr + B_offsets, mask=B_mask, other=0.0)

        accumulator += tl.dot(A_data, B_data)

        A_offsets += BLOCK_SIZE_K
        B_offsets += BLOCK_SIZE_K * N

    output_offsets = offsets_M[:, None] * N + offsets_N[None, :]
    output_mask = mask_M[:, None] & mask_N[None, :]

    tl.store(c_ptr + output_offsets + pid2 * M * N, accumulator, output_mask)


_bmm_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUPSIZE": gs,
            },
            num_warps=nw, num_stages=ns,
        )
        for bm in [32, 64, 128]
        for bn in [32, 64, 128]
        for bk in [32, 64]
        for gs in [1, 8]
        for nw in [4, 8]
        for ns in [2, 3, 4]
    ],
    key=["BATCH", "M", "N", "K"],
)(_bmm_kernel)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    C = torch.empty(BATCH * M * N, dtype=A.dtype, device=A.device)

    if autotune:
        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_SIZE_M"]),
            triton.cdiv(N, meta["BLOCK_SIZE_N"]),
            BATCH,
        )
        _bmm_kernel_autotuned[grid](A, B, C, BATCH, M, N, K)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (
            triton.cdiv(M, cfg["BLOCK_SIZE_M"]),
            triton.cdiv(N, cfg["BLOCK_SIZE_N"]),
            BATCH,
        )
        _bmm_kernel[grid](
            A, B, C, BATCH, M, N, K,
            BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
            BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
            BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"],
            GROUPSIZE=cfg["GROUPSIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return C


def get_last_config() -> dict | None:
    cfg = getattr(_bmm_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE_M": cfg.kwargs["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": cfg.kwargs["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": cfg.kwargs["BLOCK_SIZE_K"],
        "GROUPSIZE":    cfg.kwargs["GROUPSIZE"],
        "num_warps":    cfg.num_warps,
        "num_stages":   cfg.num_stages,
    }
