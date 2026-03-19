import torch
import triton
import triton.language as tl


@triton.jit
def _kl_divergence_kernel(p_ptr, q_ptr, out_ptr, n_elements, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    q = tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    p = tl.maximum(p, eps)
    q = tl.maximum(q, eps)
    out = p * (tl.log(p) - tl.log(q))
    tl.store(out_ptr + offsets, out, mask=mask)


def run(p: torch.Tensor, q: torch.Tensor, eps: float, block_size: int = 1024, **kwargs):
    if p.shape != q.shape:
        raise ValueError("Input tensors must have the same shape.")
    p = p.contiguous()
    q = q.contiguous()
    out = torch.empty(p.shape, device=p.device, dtype=torch.float32)
    n_elements = p.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _kl_divergence_kernel[grid](p, q, out, n_elements, eps, BLOCK_SIZE=block_size)
    return out
