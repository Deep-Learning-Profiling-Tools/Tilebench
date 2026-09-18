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
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0)

    # BLOCK == TILE_SIZE so each program maps to exactly one scale entry
    s = tl.load(S_ptr + pid_m * stride_sm + pid_n * stride_sn)

    y = (x.to(tl.float32) * s.to(tl.float32)).to(x.dtype.element_ty)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)

    # Lock block size to TILE_SIZE so one CTA == one scale entry
    BLOCK_M = TILE_SIZE
    BLOCK_N = TILE_SIZE
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _wdq_kernel[grid](
        X, S, Y,
        M, N,
        X.stride(0), X.stride(1),
        S.stride(0), S.stride(1),
        Y.stride(0), Y.stride(1),
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
