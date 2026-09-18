import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(
        x_ptr + offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    neg = (x.to(tl.float32) * 0.01).to(x_ptr.dtype.element_ty)
    y = tl.where(x > 0.0, x, neg)

    tl.store(
        out_ptr + offs,
        y,
        mask=mask,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _leaky_relu_kernel[grid](
        input, output, n_elements,
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
