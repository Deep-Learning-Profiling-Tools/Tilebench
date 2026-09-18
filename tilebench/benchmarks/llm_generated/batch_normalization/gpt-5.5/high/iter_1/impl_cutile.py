import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _partial_kernel(input, scratch, BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    bid_c = ct.bid(0)
    bid_n = ct.bid(1)

    x = ct.load(
        input,
        index=(bid_n, bid_c),
        shape=(BLOCK_N, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x32 = ct.astype(x, np.float32)

    s = ct.sum(x32, axis=0, keepdims=True)
    ss = ct.sum(x32 * x32, axis=0, keepdims=True)

    s3 = ct.reshape(s, (1, 1, BLOCK_C))
    ss3 = ct.reshape(ss, (1, 1, BLOCK_C))

    ct.store(scratch, index=(0, bid_n, bid_c), tile=s3)
    ct.store(scratch, index=(1, bid_n, bid_c), tile=ss3)


@ct.kernel(occupancy=4)
def _stats_kernel(scratch, gamma, beta, stats, N: ConstInt, eps: float,
                  BLOCK_P: ConstInt, BLOCK_C: ConstInt):
    bid_c = ct.bid(0)

    ps = ct.load(
        scratch,
        index=(0, 0, bid_c),
        shape=(1, BLOCK_P, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pss = ct.load(
        scratch,
        index=(1, 0, bid_c),
        shape=(1, BLOCK_P, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )

    sum_vals = ct.sum(ps, axis=1, keepdims=False)
    sumsq_vals = ct.sum(pss, axis=1, keepdims=False)

    inv_n = 1.0 / N
    mean = sum_vals * inv_n
    var = sumsq_vals * inv_n - mean * mean
    var = ct.maximum(var, 0.0)
    invstd = ct.rsqrt(var + eps)

    g = ct.load(
        gamma,
        index=(bid_c,),
        shape=(BLOCK_C,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    b = ct.load(
        beta,
        index=(bid_c,),
        shape=(BLOCK_C,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    g2 = ct.reshape(ct.astype(g, np.float32), (1, BLOCK_C))
    b2 = ct.reshape(ct.astype(b, np.float32), (1, BLOCK_C))

    scale = invstd * g2
    bias = b2 - mean * scale

    ct.store(stats, index=(0, bid_c), tile=scale)
    ct.store(stats, index=(1, bid_c), tile=bias)


@ct.kernel(occupancy=4)
def _apply_kernel(input, stats, output,
                  BLOCK_M: ConstInt, BLOCK_C: ConstInt):
    bid_c = ct.bid(0)
    bid_m = ct.bid(1)

    x = ct.load(
        input,
        index=(bid_m, bid_c),
        shape=(BLOCK_M, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x32 = ct.astype(x, np.float32)

    scale = ct.load(
        stats,
        index=(0, bid_c),
        shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        stats,
        index=(1, bid_c),
        shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )

    y = x32 * scale + bias

    ct.store(output, index=(bid_m, bid_c), tile=ct.astype(y, input.dtype))


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    output = torch.empty_like(input)

    REDUCE_BLOCK_N = 64
    REDUCE_BLOCK_C = 64
    STAT_BLOCK_P = 512
    STAT_BLOCK_C = 8
    APPLY_BLOCK_M = 16
    APPLY_BLOCK_C = 128

    partial_occupancy = 4
    stats_occupancy = 4
    apply_occupancy = 4

    num_n_blocks = (N + REDUCE_BLOCK_N - 1) // REDUCE_BLOCK_N
    num_c_reduce = (C + REDUCE_BLOCK_C - 1) // REDUCE_BLOCK_C
    num_c_stats = (C + STAT_BLOCK_C - 1) // STAT_BLOCK_C
    num_m_apply = (N + APPLY_BLOCK_M - 1) // APPLY_BLOCK_M
    num_c_apply = (C + APPLY_BLOCK_C - 1) // APPLY_BLOCK_C

    scratch = torch.empty((2, num_n_blocks, C), device=input.device, dtype=torch.float32)
    stats = torch.empty((2, C), device=input.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (num_c_reduce, num_n_blocks, 1),
        _partial_kernel,
        (input, scratch, REDUCE_BLOCK_N, REDUCE_BLOCK_C),
    )

    ct.launch(
        stream,
        (num_c_stats, 1, 1),
        _stats_kernel,
        (scratch, gamma, beta, stats, N, eps, STAT_BLOCK_P, STAT_BLOCK_C),
    )

    ct.launch(
        stream,
        (num_c_apply, num_m_apply, 1),
        _apply_kernel,
        (input, stats, output, APPLY_BLOCK_M, APPLY_BLOCK_C),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "REDUCE_BLOCK_N": REDUCE_BLOCK_N,
        "REDUCE_BLOCK_C": REDUCE_BLOCK_C,
        "STAT_BLOCK_P": STAT_BLOCK_P,
        "STAT_BLOCK_C": STAT_BLOCK_C,
        "APPLY_BLOCK_M": APPLY_BLOCK_M,
        "APPLY_BLOCK_C": APPLY_BLOCK_C,
        "partial_occupancy": partial_occupancy,
        "stats_occupancy": stats_occupancy,
        "apply_occupancy": apply_occupancy,
        "partial_grid_c_major": True,
        "apply_grid_c_major": True,
        "precompute_scale_bias": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
