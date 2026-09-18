import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    # Match reference's `A + 0` to force a real kernel launch.
    tl.store(out_ptr + offs, x, mask=mask)


def run(A: torch.Tensor, N: int, **kwargs):
    B = torch.empty_like(A)
    n_elements = A.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _copy_kernel[grid](
        A, B, n_elements,
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
    return B


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
