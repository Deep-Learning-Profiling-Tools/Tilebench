import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_forward_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    stride_x_row,
    stride_y_row,
    stride_out_row,
    ncols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE

    x_ptr += row * stride_x_row
    y_ptr += row * stride_y_row
    out_ptr += row * stride_out_row

    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < ncols

    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x * tl.sigmoid(x) * y
    tl.store(out_ptr + cols, out, mask=mask)


def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    assert x.shape == y.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    if y.stride(-1) != 1:
        y = y.contiguous()
    M, N = x.shape
    output = torch.empty_like(x)
    grid = (M, triton.cdiv(N, block_size))
    _swiglu_forward_kernel[grid](
        x,
        y,
        output,
        x.stride(0),
        y.stride(0),
        output.stride(0),
        N,
        BLOCK_SIZE=block_size,
    )
    return output
