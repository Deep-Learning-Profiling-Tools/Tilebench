import math

import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    x_ptr,
    output_ptr,
    m,
    n,
    stride_xm,
    stride_xn,
    stride_om,
    stride_on,
    BLOCK_TILE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
    offs_n = pid_n * BLOCK_TILE + tl.arange(0, BLOCK_TILE)
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    values = tl.load(x_ptrs, mask=mask)

    out_ptrs = output_ptr + offs_n[None, :] * stride_om + offs_m[:, None] * stride_on
    tl.store(out_ptrs, values, mask=mask)


def _tile_dim_from_block_size(block_size: int) -> int:
    root = int(math.sqrt(block_size))
    tile = 1
    while tile * 2 <= root:
        tile *= 2
    return tile


def run(x: torch.Tensor, block_size: int = 1024):
    m, n = x.shape
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)

    tile = _tile_dim_from_block_size(block_size)
    grid = (triton.cdiv(m, tile), triton.cdiv(n, tile))
    transpose_kernel[grid](
        x,
        output,
        m,
        n,
        x.stride(0),
        x.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_TILE=tile,
    )
    return output
