import torch
import triton
import triton.language as tl


@triton.jit
def _divergence_kernel(x_ptr, y_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    diff = x - y
    out = (diff * diff) / (y * y + eps)
    tl.store(out_ptr + offsets, out, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor, eps: float, block_size: int = 1024, **kwargs):
    del kwargs
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _divergence_kernel[grid](x, y, out, n_elements, eps, BLOCK_SIZE=block_size)
    return out
