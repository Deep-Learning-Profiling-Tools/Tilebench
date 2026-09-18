import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _partial_kernel(x_ptr, scratch_ptr, N, C, NUM_N_BLOCKS,
                    BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    offs = rows[:, None] * C + cols[None, :]
    mask = (rows[:, None] < N) & (cols[None, :] < C)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)

    s = tl.sum(x, axis=0)
    ss = tl.sum(x * x, axis=0)

    col_mask = cols < C
    out_offs = pid_n * C + cols
    plane = NUM_N_BLOCKS * C

    tl.store(scratch_ptr + out_offs, s, mask=col_mask)
    tl.store(scratch_ptr + plane + out_offs, ss, mask=col_mask)


@triton.jit
def _stats_kernel(scratch_ptr, stats_ptr, N, C, NUM_N_BLOCKS, eps,
                  BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_c = tl.program_id(0)

    parts = tl.arange(0, BLOCK_P)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    offs = parts[:, None] * C + cols[None, :]
    mask = (parts[:, None] < NUM_N_BLOCKS) & (cols[None, :] < C)

    plane = NUM_N_BLOCKS * C
    ps = tl.load(scratch_ptr + offs, mask=mask, other=0.0,
                 eviction_policy="evict_first")
    pss = tl.load(scratch_ptr + plane + offs, mask=mask, other=0.0,
                  eviction_policy="evict_first")

    sum_vals = tl.sum(ps, axis=0)
    sumsq_vals = tl.sum(pss, axis=0)

    inv_n = 1.0 / N
    mean = sum_vals * inv_n
    var = sumsq_vals * inv_n - mean * mean
    var = tl.maximum(var, 0.0)
    invstd = tl.rsqrt(var + eps)

    col_mask = cols < C
    tl.store(stats_ptr + cols, mean, mask=col_mask)
    tl.store(stats_ptr + C + cols, invstd, mask=col_mask)


@triton.jit
def _apply_kernel(x_ptr, gamma_ptr, beta_ptr, stats_ptr, out_ptr, N, C,
                  BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    offs = rows[:, None] * C + cols[None, :]
    mask = (rows[:, None] < N) & (cols[None, :] < C)
    col_mask = cols < C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                eviction_policy="evict_first").to(tl.float32)

    mean = tl.load(stats_ptr + cols, mask=col_mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    invstd = tl.load(stats_ptr + C + cols, mask=col_mask, other=0.0,
                     eviction_policy="evict_last").to(tl.float32)
    gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)
    beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)

    y = (x - mean[None, :]) * invstd[None, :]
    y = y * gamma[None, :] + beta[None, :]

    tl.store(out_ptr + offs, y, mask=mask)


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    output = torch.empty_like(input)

    REDUCE_BLOCK_N = 256
    REDUCE_BLOCK_C = 16
    STAT_BLOCK_P = 256
    APPLY_BLOCK_M = 16
    APPLY_BLOCK_C = 64

    partial_num_warps = 8
    stats_num_warps = 8
    apply_num_warps = 4
    num_stages = 4

    num_n_blocks = triton.cdiv(N, REDUCE_BLOCK_N)
    scratch = torch.empty((2, num_n_blocks, C), device=input.device, dtype=torch.float32)
    stats = torch.empty((2, C), device=input.device, dtype=torch.float32)

    grid_partial = (num_n_blocks, triton.cdiv(C, REDUCE_BLOCK_C))
    _partial_kernel[grid_partial](
        input, scratch, N, C, num_n_blocks,
        BLOCK_N=REDUCE_BLOCK_N,
        BLOCK_C=REDUCE_BLOCK_C,
        num_warps=partial_num_warps,
        num_stages=num_stages,
    )

    grid_stats = (triton.cdiv(C, REDUCE_BLOCK_C),)
    _stats_kernel[grid_stats](
        scratch, stats, N, C, num_n_blocks, eps,
        BLOCK_P=STAT_BLOCK_P,
        BLOCK_C=REDUCE_BLOCK_C,
        num_warps=stats_num_warps,
        num_stages=num_stages,
    )

    grid_apply = (triton.cdiv(N, APPLY_BLOCK_M), triton.cdiv(C, APPLY_BLOCK_C))
    _apply_kernel[grid_apply](
        input, gamma, beta, stats, output, N, C,
        BLOCK_M=APPLY_BLOCK_M,
        BLOCK_C=APPLY_BLOCK_C,
        num_warps=apply_num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "REDUCE_BLOCK_N": REDUCE_BLOCK_N,
        "REDUCE_BLOCK_C": REDUCE_BLOCK_C,
        "STAT_BLOCK_P": STAT_BLOCK_P,
        "APPLY_BLOCK_M": APPLY_BLOCK_M,
        "APPLY_BLOCK_C": APPLY_BLOCK_C,
        "partial_num_warps": partial_num_warps,
        "stats_num_warps": stats_num_warps,
        "apply_num_warps": apply_num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
