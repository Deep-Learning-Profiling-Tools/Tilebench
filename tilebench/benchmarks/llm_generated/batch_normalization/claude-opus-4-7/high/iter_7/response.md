Looking at the trajectory:
- Iter 3 was best triton (70.4%) with BLOCK_N_APPLY=16, num_warps_apply=8, num_stages=2
- Iter 6 regressed by going to BLOCK_N_APPLY=32

The bottleneck is bandwidth on the apply kernel for fp16/bf16 (62.7% roofline). I'll start from iter 3's config but try reducing apply warps to 4 (better occupancy / less register pressure per tile) and bump pipeline stages to 3 to hide HBM latency better. For cutile, I'll try a larger BLOCK_N_APPLY (16) with higher occupancy.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _stats_partial_kernel(x_ptr, psum_ptr, psumsq_ptr, N, C,
                          BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C
    ptrs = x_ptr + offs_n[:, None] * C + offs_c[None, :]
    x = tl.load(ptrs, mask=mask_n[:, None] & mask_c[None, :], other=0.0).to(tl.float32)
    sx = tl.sum(x, axis=0)
    sxx = tl.sum(x * x, axis=0)
    tl.store(psum_ptr + pid_n * C + offs_c, sx, mask=mask_c)
    tl.store(psumsq_ptr + pid_n * C + offs_c, sxx, mask=mask_c)


@triton.jit
def _finalize_kernel(psum_ptr, psumsq_ptr, gamma_ptr, beta_ptr,
                     scale_ptr, bias_ptr,
                     N, C, NUM_NB, eps,
                     BLOCK_C: tl.constexpr, BLOCK_NB: tl.constexpr):
    pid = tl.program_id(0)
    offs_c = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    sx = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sxx = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for nb_start in range(0, NUM_NB, BLOCK_NB):
        offs_nb = nb_start + tl.arange(0, BLOCK_NB)
        mask_nb = offs_nb < NUM_NB
        m = mask_nb[:, None] & mask_c[None, :]
        ptrs_s = psum_ptr + offs_nb[:, None] * C + offs_c[None, :]
        ptrs_sq = psumsq_ptr + offs_nb[:, None] * C + offs_c[None, :]
        p1 = tl.load(ptrs_s, mask=m, other=0.0)
        p2 = tl.load(ptrs_sq, mask=m, other=0.0)
        sx += tl.sum(p1, axis=0)
        sxx += tl.sum(p2, axis=0)

    inv_N = 1.0 / N
    mean = sx * inv_N
    var = sxx * inv_N - mean * mean
    rstd = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    scale = rstd * gamma
    bias = beta - mean * scale

    tl.store(scale_ptr + offs_c, scale, mask=mask_c)
    tl.store(bias_ptr + offs_c, bias, mask=mask_c)


@triton.jit
def _apply_kernel(x_ptr, out_ptr, scale_ptr, bias_ptr,
                  N, C, BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_c = tl.arange(0, BLOCK_C)
    mask_n = offs_n < N
    mask_c = offs_c < C

    scale = tl.load(scale_ptr + offs_c, mask=mask_c, other=0.0)
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    ptrs = x_ptr + offs_n[:, None] * C + offs_c[None, :]
    out_ptrs = out_ptr + offs_n[:, None] * C + offs_c[None, :]
    mask = mask_n[:, None] & mask_c[None, :]
    x = tl.load(ptrs, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)
    y = x * scale[None, :] + bias[None, :]
    tl.store(out_ptrs, y, mask=mask)


def run(input, gamma, beta, N, C, eps, **kwargs):
    output = torch.empty_like(input)

    BLOCK_N_STATS = 256
    BLOCK_C_STATS = 128
    BLOCK_N_APPLY = 16
    BLOCK_C_APPLY = triton.next_power_of_2(C)
    BLOCK_C_FIN = 64
    num_nb = triton.cdiv(N, BLOCK_N_STATS)
    BLOCK_NB_FIN = max(32, triton.next_power_of_2(num_nb))
    if BLOCK_NB_FIN > 256:
        BLOCK_NB_FIN = 256

    psum = torch.empty((num_nb, C), device=input.device, dtype=torch.float32)
    psumsq = torch.empty((num_nb, C), device=input.device, dtype=torch.float32)
    scale = torch.empty(C, device=input.device, dtype=torch.float32)
    bias = torch.empty(C, device=input.device, dtype=torch.float32)

    grid_s = (num_nb, triton.cdiv(C, BLOCK_C_STATS))
    _stats_partial_kernel[grid_s](
        input, psum, psumsq, N, C,
        BLOCK_N=BLOCK_N_STATS, BLOCK_C=BLOCK_C_STATS,
        num_warps=4, num_stages=2,
    )

    grid_f = (triton.cdiv(C, BLOCK_C_FIN),)
    _finalize_kernel[grid_f](
        psum, psumsq, gamma, beta, scale, bias,
        N, C, num_nb, eps,
        BLOCK_C=BLOCK_C_FIN, BLOCK_NB=BLOCK_NB_FIN,
        num_warps=2, num_stages=2,
    )

    grid_a = (triton.cdiv(N, BLOCK_N_APPLY),)
    _apply_kernel[grid_a](
        input, output, scale, bias, N, C,
        BLOCK_N=BLOCK_N_APPLY, BLOCK_C=BLOCK_C_APPLY,
        num_warps=4, num_stages=3,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N_STATS": BLOCK_N_STATS, "BLOCK_C_STATS": BLOCK_C_STATS,
        "BLOCK_N_APPLY": BLOCK_N_APPLY, "BLOCK_C_APPLY": BLOCK_C_APPLY,
        "BLOCK_C_FIN": BLOCK_C_FIN, "BLOCK_NB_FIN": BLOCK_NB_FIN,
        "num_warps_stats": 4, "num_warps_apply": 4, "num_stages_apply": 3,
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


@ct.kernel(occupancy=2)
def _stats_partial(x, psum, psumsq,
                   BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_n = ct.bid(0)
    pid_c = ct.bid(1)
    x_tile = ct.load(x, index=(pid_n, pid_c),
                     shape=(BLOCK_N, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_tile, np.float32)
    sx = ct.sum(x_f, axis=0, keepdims=True)
    sxx = ct.sum(x_f * x_f, axis=0, keepdims=True)
    ct.store(psum, index=(pid_n, pid_c), tile=sx)
    ct.store(psumsq, index=(pid_n, pid_c), tile=sxx)


@ct.kernel(occupancy=2)
def _finalize(psum, psumsq, gamma, beta, scale_out, bias_out, eps,
              N: ConstInt, NUM_NB: ConstInt,
              BLOCK_C: ConstInt, BLOCK_NB: ConstInt):
    pid = ct.bid(0)
    sx = ct.full((BLOCK_C,), 0.0, dtype=np.float32)
    sxx = ct.full((BLOCK_C,), 0.0, dtype=np.float32)
    num_tiles = ct.cdiv(NUM_NB, BLOCK_NB)
    for j in range(num_tiles):
        ps = ct.load(psum, index=(j, pid),
                     shape=(BLOCK_NB, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
        psq = ct.load(psumsq, index=(j, pid),
                      shape=(BLOCK_NB, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
        sx = sx + ct.sum(ps, axis=0)
        sxx = sxx + ct.sum(psq, axis=0)
    inv_N = 1.0 / N
    m = sx * inv_N
    var = sxx * inv_N - m * m
    r = ct.rsqrt(var + eps)
    g = ct.astype(ct.load(gamma, index=(pid,), shape=(BLOCK_C,)), np.float32)
    b = ct.astype(ct.load(beta, index=(pid,), shape=(BLOCK_C,)), np.float32)
    sc = r * g
    bs = b - m * sc
    ct.store(scale_out, index=(pid,), tile=sc)
    ct.store(bias_out, index=(pid,), tile=bs)


@ct.kernel(occupancy=6)
def _apply(x, out, scale_in, bias_in,
           BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    pid_n = ct.bid(0)
    scale = ct.load(scale_in, index=(0,), shape=(BLOCK_C,))
    bias = ct.load(bias_in, index=(0,), shape=(BLOCK_C,))
    x_tile = ct.load(x, index=(pid_n, 0),
                     shape=(BLOCK_N, BLOCK_C),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x_tile, np.float32)
    y = x_f * scale[None, :] + bias[None, :]
    y_out = ct.astype(y, x.dtype)
    ct.store(out, index=(pid_n, 0), tile=y_out)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def run(input, gamma, beta, N, C, eps, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BLOCK_N_STATS = 256
    BLOCK_C_STATS = 128
    BLOCK_N_APPLY = 8
    BLOCK_C_APPLY = _next_pow2(C)
    BLOCK_C_FIN = 64
    num_nb = _cdiv(N, BLOCK_N_STATS)
    BLOCK_NB_FIN = max(32, _next_pow2(num_nb))
    if BLOCK_NB_FIN > 256:
        BLOCK_NB_FIN = 256

    psum = torch.empty((num_nb, C), device=input.device, dtype=torch.float32)
    psumsq = torch.empty((num_nb, C), device=input.device, dtype=torch.float32)
    scale = torch.empty(C, device=input.device, dtype=torch.float32)
    bias_buf = torch.empty(C, device=input.device, dtype=torch.float32)

    grid_s = (num_nb, _cdiv(C, BLOCK_C_STATS), 1)
    ct.launch(stream, grid_s, _stats_partial,
              (input, psum, psumsq, BLOCK_N_STATS, BLOCK_C_STATS))

    grid_f = (_cdiv(C, BLOCK_C_FIN), 1, 1)
    ct.launch(stream, grid_f, _finalize,
              (psum, psumsq, gamma, beta, scale, bias_buf, float(eps),
               int(N), int(num_nb), BLOCK_C_FIN, BLOCK_NB_FIN))

    grid_a = (_cdiv(N, BLOCK_N_APPLY), 1, 1)
    ct.launch(stream, grid_a, _apply,
              (input, output, scale, bias_buf,
               BLOCK_N_APPLY, BLOCK_C_APPLY))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N_STATS": BLOCK_N_STATS, "BLOCK_C_STATS": BLOCK_C_STATS,
        "BLOCK_N_APPLY": BLOCK_N_APPLY, "BLOCK_C_APPLY": BLOCK_C_APPLY,
        "BLOCK_C_FIN": BLOCK_C_FIN, "BLOCK_NB_FIN": BLOCK_NB_FIN,
        "occupancy_stats": 2, "occupancy_finalize": 2, "occupancy_apply": 6,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted triton to iter 3's structure (BLOCK_N_APPLY=16) and tweaked apply to `num_warps=4, num_stages=3` for better pipelining/occupancy on the bandwidth-bound apply pass. For cutile, kept iter 6's apply config (BLOCK_N=8) but bumped `occupancy_apply` to 6 for more parallel CTAs on the apply pass.
