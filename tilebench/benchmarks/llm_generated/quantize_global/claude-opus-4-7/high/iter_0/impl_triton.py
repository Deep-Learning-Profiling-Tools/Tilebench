import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _quantize_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x.to(tl.float16), mask=mask)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    output = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    n_elements = x.numel()

    BLOCK_SIZE = 4096
    num_warps = 4
    num_stages = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _quantize_kernel[grid](
        x, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_SIZE": BLOCK_SIZE,
                      "num_warps": num_warps,
                      "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
