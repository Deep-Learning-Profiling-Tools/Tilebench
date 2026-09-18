import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matrix_copy_kernel(A_ptr, B_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(A_ptr + offsets, mask=mask, other=0)
    tl.store(B_ptr + offsets, x, mask=mask)


def run(A: torch.Tensor, N: int, **kwargs):
    B = torch.empty_like(A)
    n_elements = A.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 1

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _matrix_copy_kernel[grid](
        A,
        B,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_SIZE": BLOCK_SIZE,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return B


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
