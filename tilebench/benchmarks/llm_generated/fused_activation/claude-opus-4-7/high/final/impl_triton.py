import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _fused_act_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    z = x * g + b
    y = z / (1.0 + tl.exp(-z))
    tl.store(out_ptr + offs, y, mask=mask)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    output = torch.empty_like(x)
    n = x.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _fused_act_kernel[grid](
        x, gate, bias, output, n,
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
