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
    # Interleave in registers: result has shape (2*BLOCK,) where
    # result[2i] = a[i], result[2i+1] = b[i]
    interleaved = tl.interleave(a, b)
    out_offs = pid * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
    out_mask = out_offs < (2 * N)
    tl.store(out_ptr + out_offs, interleaved, mask=out_mask)


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)

    BLOCK = 4096
    num_warps = 8
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
