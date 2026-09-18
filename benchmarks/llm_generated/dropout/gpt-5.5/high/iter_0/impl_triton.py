import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _dropout_kernel(x_ptr, keep_ptr, out_ptr, n_elements, scale, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first")
    keep = tl.load(keep_ptr + offsets, mask=mask, other=0, eviction_policy="evict_first")

    y = x * scale
    out = tl.where(keep != 0, y, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 2048
    num_warps = 4
    num_stages = 2

    scale = 1.0 / (1.0 - float(p))

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _dropout_kernel[grid](
        x,
        x_keep,
        output,
        n_elements,
        scale,
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
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
