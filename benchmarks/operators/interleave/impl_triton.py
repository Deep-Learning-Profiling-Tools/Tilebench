import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 128, "num_warps": 4, "num_stages": 2}


@triton.jit
def interleave_kernel(A_ptr, B_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    a_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    b_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    a_mask = a_offsets < N
    b_mask = b_offsets < N

    a_local = tl.load(A_ptr + a_offsets, a_mask)
    b_local = tl.load(B_ptr + b_offsets, b_mask)

    output_local = tl.interleave(a_local, b_local)

    output_offsets = pid * 2 * BLOCK_SIZE + tl.arange(0, 2 * BLOCK_SIZE)

    tl.store(output_ptr + output_offsets, output_local, output_offsets < 2 * N)


_interleave_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [64, 128, 256, 512]
        for nw in [4, 8]
    ],
    key=["N"],
)(interleave_kernel)


def run(A: torch.Tensor, B: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)

    if autotune:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        _interleave_kernel_autotuned[grid](A, B, output, N)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(N, cfg["BLOCK_SIZE"]),)
        interleave_kernel[grid](
            A, B, output, N,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_interleave_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
