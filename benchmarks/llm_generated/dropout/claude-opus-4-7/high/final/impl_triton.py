import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _dropout_kernel(x_ptr, mask_ptr, out_ptr, scale, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    k = tl.load(mask_ptr + offs, mask=mask)
    keep = k != 0
    y = tl.where(keep, x.to(tl.float32) * scale, 0.0).to(x.dtype)
    tl.store(out_ptr + offs, y, mask=mask)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = 1.0 / (1.0 - p)

    BLOCK_SIZE = 4096
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _dropout_kernel[grid](
        x, x_keep, output, scale, n_elements,
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
