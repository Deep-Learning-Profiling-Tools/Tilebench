import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _transpose_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_xm,
    stride_xn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]

    if EVEN:
        tile = tl.load(x_ptrs)
        tl.store(out_ptrs, tl.trans(tile))
    else:
        load_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tile = tl.load(x_ptrs, mask=load_mask, other=0)

        store_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
        tl.store(out_ptrs, tl.trans(tile), mask=store_mask)


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")

    m = x.shape[0]
    n = x.shape[1]
    output = torch.empty((n, m), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 8
    num_stages = 3
    EVEN = (m % BLOCK_M == 0) and (n % BLOCK_N == 0)

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))
    _transpose_kernel[grid](
        x,
        output,
        m,
        n,
        x.stride(0),
        x.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        EVEN=EVEN,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "EVEN": EVEN,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
