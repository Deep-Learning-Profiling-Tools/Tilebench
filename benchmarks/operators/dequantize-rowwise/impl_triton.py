import torch
import triton
import triton.language as tl


@triton.jit
def _unpack_to_fp32_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x.to(tl.float32), mask=mask)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    x = x.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _unpack_to_fp32_kernel[grid](x, out, n_elements, BLOCK_SIZE=block_size)
    return out
