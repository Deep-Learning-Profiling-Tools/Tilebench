import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _wdq_kernel(X_ptr, S_ptr, Y_ptr,
                M, N,
                stride_xm, stride_xn,
                stride_sm, stride_sn,
                stride_ym, stride_yn,
                M_S, N_S,
                TILE_SIZE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_x = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_x, other=0)

    si = offs_m // TILE_SIZE
    sj = offs_n // TILE_SIZE
    mask_s = (si[:, None] < M_S) & (sj[None, :] < N_S)
    s_ptrs = S_ptr + si[:, None] * stride_sm + sj[None, :] * stride_sn
    s = tl.load(s_ptrs, mask=mask_s, other=0)

    y = x.to(tl.float32) * s.to(tl.float32)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_x)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)

    # Smaller per-program tile area than iter 2 (was 128×256) to reduce
    # register pressure for fp32 (the worst-performing dtype). Same total
    # work but more in-flight programs to hide HBM latency.
    BLOCK_M = 64
    BLOCK_N = 256
    num_warps = 8
    num_stages = 4

    M_S = (M + TILE_SIZE - 1) // TILE_SIZE
    N_S = (N + TILE_SIZE - 1) // TILE_SIZE

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _wdq_kernel[grid](
        X, S, Y,
        M, N,
        X.stride(0), X.stride(1),
        S.stride(0), S.stride(1),
        Y.stride(0), Y.stride(1),
        M_S, N_S,
        TILE_SIZE=TILE_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "num_warps": num_warps, "num_stages": num_stages,
    })
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
