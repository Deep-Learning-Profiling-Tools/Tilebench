import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _interleave_kernel(A_ptr, B_ptr, out_ptr, N,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(A_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")
    b = tl.load(B_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")

    vals = tl.interleave(a, b)

    out_offs = pid * (2 * BLOCK_SIZE) + tl.arange(0, 2 * BLOCK_SIZE)
    out_mask = out_offs < (2 * N)
    tl.store(out_ptr + out_offs, vals, mask=out_mask, eviction_policy="evict_first")


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty((2 * N,), dtype=A.dtype, device=A.device)

    BLOCK_SIZE = 2048
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _interleave_kernel[grid](
        A, B, output, N,
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
