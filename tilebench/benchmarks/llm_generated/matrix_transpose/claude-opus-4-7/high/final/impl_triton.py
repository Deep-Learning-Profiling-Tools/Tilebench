import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _transpose_kernel(x_ptr, out_ptr, M, N,
                      stride_xm, stride_xn,
                      stride_om, stride_on,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    in_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=in_mask)

    x_t = tl.trans(x)  # [BN, BM]

    o_ptrs = out_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(o_ptrs, x_t, mask=out_mask)


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    M, N = x.shape
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_kernel[grid](
        x, output, M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                      "num_warps": num_warps, "num_stages": num_stages})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
