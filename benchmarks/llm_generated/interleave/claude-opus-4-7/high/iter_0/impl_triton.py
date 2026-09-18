import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _interleave_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out_offs = 2 * offs
    tl.store(out_ptr + out_offs, a, mask=mask)
    tl.store(out_ptr + out_offs + 1, b, mask=mask)


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)

    BLOCK = 2048
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK),)
    _interleave_kernel[grid](
        A, B, output, N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK": BLOCK, "num_warps": num_warps, "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
