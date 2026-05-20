import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2}


@triton.jit
def matrix_copy_kernel(
    A_ptr, B_ptr,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    a_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[N, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    b_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[N, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )

    x = a_desc.load([start_m, start_n])
    b_desc.store([start_m, start_n], x)


_matrix_copy_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bm in [16, 32, 64]
        for bn in [16, 32, 64]
        for nw in [4, 8]
        for ns in [1, 2]
    ],
    key=["N"],
)(matrix_copy_kernel)


def run(A: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    B = torch.empty_like(A)
    ensure_tma_available()
    if not A.is_contiguous():
        A = A.contiguous()

    total = N * N

    if autotune:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _matrix_copy_kernel_autotuned[grid](A, B, N)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(N, cfg["BLOCK_M"]), triton.cdiv(N, cfg["BLOCK_N"]))
        matrix_copy_kernel[grid](
            A, B, N,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_N=cfg["BLOCK_N"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return B


def get_last_config() -> dict | None:
    cfg = getattr(_matrix_copy_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_M": cfg.kwargs["BLOCK_M"], "BLOCK_N": cfg.kwargs["BLOCK_N"], "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
