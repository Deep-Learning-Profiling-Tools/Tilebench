import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.where(x >= 0, x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


def run(x: torch.Tensor, block_size: int = 1024):
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    relu_kernel[grid](x, output, n_elements, BLOCK_SIZE=block_size)
    return output
