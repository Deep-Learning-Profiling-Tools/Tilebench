Looking at this batch_normalization op: input [N, C], compute mean/var over N per channel, then apply gamma*normalized+beta. I'll use a 2-kernel pipeline: (1) stats kernel computing mean+rstd per channel via serial N reduction, (2) apply kernel doing the element-wise normalization with broadcasting.

```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _stats_kernel(x, mean_out, rstd_out, eps,
                  N: ConstInt, BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_c = ct.bid(0)

    sum_x = ct.full((BLOCK_C,), 0.0, dtype=np.float32)
    sum_xx = ct.full((BLOCK_C,), 0.0, dtype=np.float32)

    num_n_tiles = ct.cdiv(N, BLOCK_N)
    for n_tile in range(0, num_n_tiles):
        x_tile = ct.load(x, index=(n_tile, pid_c),
                         shape=(BLOCK_N, BLOCK_C),
                         padding_mode=ct.PaddingMode.ZERO)
        x_f = ct.astype(x_tile, np.float32)
        sum_x = sum_x + ct.sum(x_f, axis=0)
        sum_xx = sum_xx + ct.sum(x_f * x_f, axis=0)

    inv_N = 1.0 / N
    mean = sum_x * inv_N
    var = sum_xx * inv_N - mean * mean
    rstd = ct.rsqrt(var + eps)

    ct.store(mean_out, index=(pid_c,), tile=mean)
    ct.store(rstd_out, index=(pid_c,), tile=rstd)


@ct.kernel
def _apply_kernel(x, out, mean_in, rstd_in, gamma, beta,
                  BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_n = ct.bid(0)
    pid_c = ct.bid(1)

    mean = ct.load(mean_in, index=(pid_c,), shape=(BLOCK_C,))
    rstd = ct.load(rstd_in, index=(pid_c,), shape=(BLOCK_C,))
    gamma_t = ct.astype(ct.load(gamma, index=(pid_c,), shape=(BLOCK_C,)), np.float32)
    beta_t = ct.astype(ct.load(beta, index=(pid_c,), shape=(BLOCK_C,)), np.float32)

    x_tile = ct.load(x, index=(pid_n, pid_c),
                     shape=(BLOCK_N, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_tile, np.float32)

    y = (x_f - mean[None, :]) * rstd[None, :] * gamma_t[None, :] + beta_t[None, :]
    y_out = ct.astype(y, x.dtype)
    ct.store(out, index=(pid_n, pid_c), tile=y_out)


def run(input, gamma, beta, N, C, eps, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_N_STATS = 256
    BLOCK_C_STATS = 64
    BLOCK_N_APPLY = 64
    BLOCK_C_APPLY = 128
    occupancy_stats = 2
    occupancy_apply = 4

    mean = torch.empty(C, device=input.device, dtype=torch.float32)
    rstd = torch.empty(C, device=input.device, dtype=torch.float32)

    grid_s = (ct.cdiv(C, BLOCK_C_STATS), 1, 1)
    stats_kernel = _stats_kernel.with_hints(occupancy=occupancy_stats)
    ct.launch(stream, grid_s, stats_kernel,
              (input, mean, rstd, float(eps), int(N),
               BLOCK_N_STATS, BLOCK_C_STATS))

    grid_a = (ct.cdiv(N, BLOCK_N_APPLY), ct.cdiv(C, BLOCK_C_APPLY), 1)
    apply_kernel = _apply_kernel.with_hints(occupancy=occupancy_apply)
    ct.launch(stream, grid_a, apply_kernel,
              (input, output, mean, rstd, gamma, beta,
               BLOCK_N_APPLY, BLOCK_C_APPLY))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N_STATS": BLOCK_N_STATS, "BLOCK_C_STATS": BLOCK_C_STATS,
        "BLOCK_N_APPLY": BLOCK_N_APPLY, "BLOCK_C_APPLY": BLOCK_C_APPLY,
        "occupancy_stats": occupancy_stats, "occupancy_apply": occupancy_apply,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
