import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _global_barrier(barrier_ptr, PHASE: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    old = tl.atomic_add(barrier_ptr, 1, sem="acq_rel", scope="gpu")
    if old == NUM_BLOCKS - 1:
        tl.atomic_xchg(barrier_ptr, 0, sem="release", scope="gpu")
        tl.atomic_xchg(barrier_ptr + 1, PHASE, sem="release", scope="gpu")
    else:
        while tl.load(barrier_ptr + 1, volatile=True, cache_modifier=".cv") < PHASE:
            pass


@triton.jit
def _bn_persistent_kernel(x_ptr, gamma_ptr, beta_ptr, out_ptr,
                          stats_ptr, barrier_ptr,
                          N, C, eps,
                          REDUCE_BLOCK_N: tl.constexpr,
                          REDUCE_BLOCK_C: tl.constexpr,
                          APPLY_BLOCK_M: tl.constexpr,
                          APPLY_BLOCK_C: tl.constexpr,
                          NUM_PERSISTENT: tl.constexpr,
                          LOOP_NUM_STAGES: tl.constexpr):
    pid = tl.program_id(0)

    num_n_blocks = tl.cdiv(N, REDUCE_BLOCK_N)
    num_c_reduce = tl.cdiv(C, REDUCE_BLOCK_C)
    num_m_apply = tl.cdiv(N, APPLY_BLOCK_M)
    num_c_apply = tl.cdiv(C, APPLY_BLOCK_C)

    # Initialize per-channel sum / sumsq buffers.
    r_cols_base = tl.arange(0, REDUCE_BLOCK_C)
    for cblk in tl.range(pid, num_c_reduce, NUM_PERSISTENT, num_stages=1):
        cols = cblk * REDUCE_BLOCK_C + r_cols_base
        col_mask = cols < C
        tl.store(stats_ptr + cols, 0.0, mask=col_mask)
        tl.store(stats_ptr + C + cols, 0.0, mask=col_mask)

    _global_barrier(barrier_ptr, 1, NUM_PERSISTENT)

    # Persistent parallel partial reduction.  We use atomics into the 2*C stats
    # buffer, then a grid-wide resident-block barrier before finalizing stats.
    rows_r = tl.arange(0, REDUCE_BLOCK_N)
    total_reduce_tiles = num_n_blocks * num_c_reduce

    for tile in tl.range(pid, total_reduce_tiles, NUM_PERSISTENT,
                         num_stages=LOOP_NUM_STAGES):
        bid_n = tile // num_c_reduce
        bid_c = tile - bid_n * num_c_reduce

        rows = bid_n * REDUCE_BLOCK_N + rows_r
        cols = bid_c * REDUCE_BLOCK_C + r_cols_base

        offs = rows[:, None] * C + cols[None, :]
        mask = (rows[:, None] < N) & (cols[None, :] < C)

        x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)

        s = tl.sum(x, axis=0)
        ss = tl.sum(x * x, axis=0)

        col_mask = cols < C
        tl.atomic_add(stats_ptr + cols, s, sem="relaxed", scope="gpu", mask=col_mask)
        tl.atomic_add(stats_ptr + C + cols, ss, sem="relaxed", scope="gpu", mask=col_mask)

    _global_barrier(barrier_ptr, 2, NUM_PERSISTENT)

    # Convert sum/sumsq into precomputed scale and bias:
    # y = x * (gamma * invstd) + (beta - mean * gamma * invstd)
    for cblk in tl.range(pid, num_c_reduce, NUM_PERSISTENT, num_stages=1):
        cols = cblk * REDUCE_BLOCK_C + r_cols_base
        col_mask = cols < C

        sum_vals = tl.load(stats_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        sumsq_vals = tl.load(stats_ptr + C + cols, mask=col_mask, other=0.0).to(tl.float32)

        inv_n = 1.0 / N
        mean = sum_vals * inv_n
        var = sumsq_vals * inv_n - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        gamma = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0,
                        eviction_policy="evict_last").to(tl.float32)
        beta = tl.load(beta_ptr + cols, mask=col_mask, other=0.0,
                       eviction_policy="evict_last").to(tl.float32)

        scale = invstd * gamma
        bias = beta - mean * scale

        tl.store(stats_ptr + cols, scale, mask=col_mask)
        tl.store(stats_ptr + C + cols, bias, mask=col_mask)

    _global_barrier(barrier_ptr, 3, NUM_PERSISTENT)

    # Persistent apply pass.  Linear tile order is C-fast for contiguous memory.
    rows_a = tl.arange(0, APPLY_BLOCK_M)
    cols_a = tl.arange(0, APPLY_BLOCK_C)
    total_apply_tiles = num_m_apply * num_c_apply

    for tile in tl.range(pid, total_apply_tiles, NUM_PERSISTENT,
                         num_stages=LOOP_NUM_STAGES):
        bid_m = tile // num_c_apply
        bid_c = tile - bid_m * num_c_apply

        rows = bid_m * APPLY_BLOCK_M + rows_a
        cols = bid_c * APPLY_BLOCK_C + cols_a

        offs = rows[:, None] * C + cols[None, :]
        mask = (rows[:, None] < N) & (cols[None, :] < C)
        col_mask = cols < C

        x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)

        scale = tl.load(stats_ptr + cols, mask=col_mask, other=0.0,
                        eviction_policy="evict_last").to(tl.float32)
        bias = tl.load(stats_ptr + C + cols, mask=col_mask, other=0.0,
                       eviction_policy="evict_last").to(tl.float32)

        y = x * scale[None, :] + bias[None, :]
        tl.store(out_ptr + offs, y, mask=mask, cache_modifier=".cs")


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    output = torch.empty_like(input)

    REDUCE_BLOCK_N = 256
    REDUCE_BLOCK_C = 16
    APPLY_BLOCK_M = 32
    APPLY_BLOCK_C = 64
    NUM_PERSISTENT = 128
    LOOP_NUM_STAGES = 3
    num_warps = 8
    num_stages = 4

    stats = torch.empty((2, C), device=input.device, dtype=torch.float32)
    barrier = torch.zeros((2,), device=input.device, dtype=torch.int32)

    _bn_persistent_kernel[(NUM_PERSISTENT,)](
        input, gamma, beta, output, stats, barrier,
        N, C, eps,
        REDUCE_BLOCK_N=REDUCE_BLOCK_N,
        REDUCE_BLOCK_C=REDUCE_BLOCK_C,
        APPLY_BLOCK_M=APPLY_BLOCK_M,
        APPLY_BLOCK_C=APPLY_BLOCK_C,
        NUM_PERSISTENT=NUM_PERSISTENT,
        LOOP_NUM_STAGES=LOOP_NUM_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "single_kernel_persistent": True,
        "REDUCE_BLOCK_N": REDUCE_BLOCK_N,
        "REDUCE_BLOCK_C": REDUCE_BLOCK_C,
        "APPLY_BLOCK_M": APPLY_BLOCK_M,
        "APPLY_BLOCK_C": APPLY_BLOCK_C,
        "NUM_PERSISTENT": NUM_PERSISTENT,
        "LOOP_NUM_STAGES": LOOP_NUM_STAGES,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "precompute_scale_bias": True,
        "atomic_stats": True,
        "apply_c_fast_linear": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
