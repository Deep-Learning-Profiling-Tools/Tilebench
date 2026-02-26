import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kernel(x_ptr, gate_ptr, bias_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x * gate + bias
    out = tl.where(z > 0, z, 0)
    tl.store(out_ptr + offsets, out, mask=mask)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, block_size: int = 1024, **kwargs):
    del kwargs
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")
    x = x.contiguous()
    gate = gate.contiguous()
    bias = bias.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _fused_kernel[grid](x, gate, bias, out, n_elements, BLOCK_SIZE=block_size)
    return out
