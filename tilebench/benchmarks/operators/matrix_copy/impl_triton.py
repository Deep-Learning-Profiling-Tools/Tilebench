import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def matrix_copy_kernel(
    A_ptr, B_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * N
    x = tl.load(A_ptr + offsets, mask=mask)
    tl.store(B_ptr + offsets, x, mask=mask)


_matrix_copy_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [1024, 2048, 4096]
        for nw in [4, 8]
        for ns in [1, 2]
    ],
    key=["N"],
)(matrix_copy_kernel)


def run(A: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    B = torch.empty_like(A)
    total = N * N

    if autotune:
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)
        _matrix_copy_kernel_autotuned[grid](A, B, N)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(total, cfg["BLOCK_SIZE"]),)
        matrix_copy_kernel[grid](
            A, B, N,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return B


def get_last_config() -> dict | None:
    cfg = getattr(_matrix_copy_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
