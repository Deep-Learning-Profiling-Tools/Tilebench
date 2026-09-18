import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _stats_kernel(x_ptr, mean_ptr, rstd_ptr, N, C, eps,
                  BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_c = tl.program_id(0)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    sum_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_xx = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        ptrs = x_ptr + offs_n[:, None] * C + offs_c[None, :]
        x = tl.load(ptrs, mask=mask_n[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_xx += tl.sum(x * x, axis=0)

    inv_N = 1.0 / N
    mean = sum_x * inv_N
    var = sum_xx * inv_N - mean * mean
    rstd = tl.rsqrt(var + eps)

    tl.store(mean_ptr + offs_c, mean, mask=mask_c)
    tl.store(rstd_ptr + offs_c, rstd, mask=mask_c)


@triton.jit
def _apply_kernel(x_ptr, out_ptr, mean_ptr, rstd_ptr, gamma_ptr, beta_ptr,
                  N, C, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C

    mean = tl.load(mean_ptr + offs_c, mask=mask_c, other=0.0)
    rstd = tl.load(rstd_ptr + offs_c, mask=mask_c, other=0.0)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)

    ptrs = x_ptr + offs_n[:, None] * C + offs_c[None, :]
    out_ptrs = out_ptr + offs_n[:, None] * C + offs_c[None, :]
    mask = mask_n[:, None] & mask_c[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean[None, :]) * rstd[None, :] * gamma[None, :] + beta[None, :]
    tl.store(out_ptrs, y, mask=mask)


def run(input, gamma, beta, N, C, eps, **kwargs):
    output = torch.empty_like(input)

    BLOCK_N_STATS = 256
    BLOCK_C_STATS = 64
    BLOCK_N_APPLY = 64
    BLOCK_C_APPLY = 128
    num_warps_stats = 4
    num_warps_apply = 4
    num_stages = 2

    mean = torch.empty(C, device=input.device, dtype=torch.float32)
    rstd = torch.empty(C, device=input.device, dtype=torch.float32)

    grid_s = (triton.cdiv(C, BLOCK_C_STATS),)
    _stats_kernel[grid_s](
        input, mean, rstd, N, C, eps,
        BLOCK_N=BLOCK_N_STATS, BLOCK_C=BLOCK_C_STATS,
        num_warps=num_warps_stats, num_stages=num_stages,
    )

    grid_a = (triton.cdiv(N, BLOCK_N_APPLY), triton.cdiv(C, BLOCK_C_APPLY))
    _apply_kernel[grid_a](
        input, output, mean, rstd, gamma, beta, N, C,
        BLOCK_N=BLOCK_N_APPLY, BLOCK_C=BLOCK_C_APPLY,
        num_warps=num_warps_apply, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N_STATS": BLOCK_N_STATS, "BLOCK_C_STATS": BLOCK_C_STATS,
        "BLOCK_N_APPLY": BLOCK_N_APPLY, "BLOCK_C_APPLY": BLOCK_C_APPLY,
        "num_warps_stats": num_warps_stats, "num_warps_apply": num_warps_apply,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
