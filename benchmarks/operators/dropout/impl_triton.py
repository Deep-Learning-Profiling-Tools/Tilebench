import torch
import triton
import triton.language as tl


@triton.jit
def _dropout_kernel(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep.to(tl.int1), x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def run(x, x_keep, p, block_size=1024):
    output = torch.empty_like(x)
    n_elements = x.numel()

    def grid(meta):
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    _dropout_kernel[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=block_size)
    return output
