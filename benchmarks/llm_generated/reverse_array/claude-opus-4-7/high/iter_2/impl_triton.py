import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Forward-stride load from the "mirror" position, then flip locally.
    # This guarantees contiguous stride-+1 access on both load and store sides,
    # which the compiler can vectorize cleanly.
    src_base = N - (pid + 1) * BLOCK_SIZE
    src_offs = src_base + tl.arange(0, BLOCK_SIZE)
    src_mask = src_offs >= 0
    safe_src = tl.maximum(src_offs, 0)
    x = tl.load(in_ptr + safe_src, mask=src_mask, other=0)
    x = tl.flip(x, 0)
    out_mask = out_offs < N
    tl.store(out_ptr + out_offs, x, mask=out_mask)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_SIZE = 8192
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _reverse_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
