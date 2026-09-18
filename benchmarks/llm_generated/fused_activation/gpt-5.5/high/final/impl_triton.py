import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _fused_activation_kernel(x_ptr, gate_ptr, bias_ptr, out_ptr, n_elements,
                             BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    z = x * gate + bias
    sigmoid = 1.0 / (1.0 + tl.exp2(-z * 1.4426950408889634))
    y = z * sigmoid

    tl.store(out_ptr + offsets, y, mask=mask)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 1024
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _fused_activation_kernel[grid](
        x, gate, bias, output, n_elements,
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
