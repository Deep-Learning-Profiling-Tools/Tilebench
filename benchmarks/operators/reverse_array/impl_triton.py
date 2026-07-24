import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def reverse_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    """
    Out-of-place reverse matching the cuTile backend's method:
      output[i] = input[N - 1 - i]
    Each program handles BLOCK_SIZE consecutive output positions. Boundary
    tiles are masked; the in-place swap variant is intentionally avoided
    because repeated kernel launches (autotune / benchmark loops) would
    cancel out the reversal.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    rev_offsets = N - 1 - offsets
    vals = tl.load(input_ptr + rev_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, vals, mask=mask)


_reverse_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [1024, 2048, 4096, 8192]
        for nw in [4, 8]
        for ns in [1, 2]

    ],
    key=["N"],
)(reverse_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output = torch.empty_like(input)

    if autotune:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE"]),)
        _reverse_kernel_autotuned[grid](input, output, N)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(N, cfg["BLOCK_SIZE"]),)
        reverse_kernel[grid](
            input, output, N,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_reverse_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
