import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

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

    start_m = pid0 * BLOCK_SIZE_M
    start_n = pid1 * BLOCK_SIZE_N

    a_desc = tl.make_tensor_descriptor(
        a_ptr + pid2 * M * K,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr + pid2 * K * N,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr + pid2 * M * N,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    for current_k_index in range(0, K, BLOCK_SIZE_K):
        A_data = a_desc.load([start_m, current_k_index])
        B_data = b_desc.load([current_k_index, start_n])
        accumulator += tl.dot(A_data, B_data, input_precision="tf32")

    c_desc.store([start_m, start_n], accumulator)


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
    ensure_tma_available()
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()

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
