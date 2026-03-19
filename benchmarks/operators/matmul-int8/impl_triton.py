import math

import torch
import triton
import triton.language as tl


@triton.jit
def _int8_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    out_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(k, BLOCK_K)):
        k_offs = k_start * BLOCK_K + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < m) & (k_offs[None, :] < k)
        b_mask = (k_offs[:, None] < k) & (offs_n[None, :] < n)
        a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.float32)
        b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.float32)
        acc += tl.dot(a, b)

    out = acc * out_scale
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptrs, out, mask=c_mask)


def _tile_dim_from_block_size(block_size: int) -> int:
    root = int(math.sqrt(block_size))
    tile = 16
    while tile * 2 <= root:
        tile *= 2
    return max(16, tile)


def run(a_q: torch.Tensor, b_q: torch.Tensor, scale: float, block_size: int = 1024, **kwargs):
    if a_q.dim() != 2 or b_q.dim() != 2:
        raise ValueError("matmul-int8 expects 2D inputs.")
    if a_q.shape[1] != b_q.shape[0]:
        raise ValueError("Inner dimensions must match for GEMM.")

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    m, k = a_q.shape
    _, n = b_q.shape
    out = torch.empty((m, n), device=a_q.device, dtype=torch.float32)

    tile = _tile_dim_from_block_size(block_size)
    grid = (triton.cdiv(m, tile), triton.cdiv(n, tile))
    _int8_matmul_kernel[grid](
        a_q,
        b_q,
        out,
        m,
        n,
        k,
        a_q.stride(0),
        a_q.stride(1),
        b_q.stride(0),
        b_q.stride(1),
        out.stride(0),
        out.stride(1),
        scale * scale,
        BLOCK_M=tile,
        BLOCK_N=tile,
        BLOCK_K=32,
    )
    return out
