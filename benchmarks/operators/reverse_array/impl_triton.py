import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


# Original LeetGPU in-place swap kernel (unchanged)
@triton.jit
def reverse_kernel(input, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    half = N // 2
    mask = offsets < half
    rev_offsets = N - 1 - offsets
    a = tl.load(input + offsets, mask=mask)
    b = tl.load(input + rev_offsets, mask=mask)

    tl.store(input + offsets, b, mask=mask)
    tl.store(input + rev_offsets, a, mask=mask)


# Out-of-place variant for autotune (in-place kernel is unsafe under
# repeated runs because swapping twice restores the original order).
@triton.jit
def _reverse_kernel_oop(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    rev_offsets = N - 1 - offsets
    vals = tl.load(input_ptr + rev_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, vals, mask=mask)


_reverse_kernel_oop_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048]
        for nw in [4, 8]
    ],
    key=["N"],
)(_reverse_kernel_oop)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    if autotune:
        output = torch.empty_like(input)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        _reverse_kernel_oop_autotuned[grid](input, output, N)
        return output
    else:
        output = input.clone()
        half = N // 2
        if half <= 0:
            return output
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(half, cfg["BLOCK_SIZE"]),)
        reverse_kernel[grid](
            output, N,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
        return output


def get_last_config() -> dict | None:
    cfg = getattr(_reverse_kernel_oop_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
