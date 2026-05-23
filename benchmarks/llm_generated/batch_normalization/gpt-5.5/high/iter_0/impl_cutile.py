import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _partial_kernel(input, scratch, BLOCK_N: ConstInt, BLOCK_C: ConstInt):
    bid_n = ct.bid(0)
    bid_c = ct.bid(1)

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


@ct.kernel
def _stats_kernel(scratch, stats, N: ConstInt, eps: float,
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

    ct.store(stats, index=(0, bid_c), tile=mean)
    ct.store(stats, index=(1, bid_c), tile=invstd)


@ct.kernel
def _apply_kernel(input, gamma, beta, stats, output,
                  BLOCK_M: ConstInt, BLOCK_C: ConstInt):
    bid_m = ct.bid(0)
    bid_c = ct.bid(1)

    x = ct.load(
        input,
        index=(bid_m, bid_c),
        shape=(BLOCK_M, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x32 = ct.astype(x, np.float32)

    mean = ct.load(
        stats,
        index=(0, bid_c),
        shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    invstd = ct.load(
        stats,
        index=(1, bid_c),
        shape=(1, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )

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

    y = (x32 - mean) * invstd
    y = y * g2 + b2

    ct.store(output, index=(bid_m, bid_c), tile=ct.astype(y, input.dtype))


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    output = torch.empty_like(input)

    REDUCE_BLOCK_N = 256
    REDUCE_BLOCK_C = 16
    STAT_BLOCK_P = 256
    APPLY_BLOCK_M = 16
    APPLY_BLOCK_C = 64

    partial_occupancy = 4
    stats_occupancy = 4
    apply_occupancy = 8

    num_n_blocks = (N + REDUCE_BLOCK_N - 1) // REDUCE_BLOCK_N
    num_c_reduce = (C + REDUCE_BLOCK_C - 1) // REDUCE_BLOCK_C
    num_m_apply = (N + APPLY_BLOCK_M - 1) // APPLY_BLOCK_M
    num_c_apply = (C + APPLY_BLOCK_C - 1) // APPLY_BLOCK_C

    scratch = torch.empty((2, num_n_blocks, C), device=input.device, dtype=torch.float32)
    stats = torch.empty((2, C), device=input.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()

    partial_kernel = _partial_kernel.with_hints(occupancy=partial_occupancy)
    ct.launch(
        stream,
        (num_n_blocks, num_c_reduce, 1),
        partial_kernel,
        (input, scratch, REDUCE_BLOCK_N, REDUCE_BLOCK_C),
    )

    stats_kernel = _stats_kernel.with_hints(occupancy=stats_occupancy)
    ct.launch(
        stream,
        (num_c_reduce, 1, 1),
        stats_kernel,
        (scratch, stats, N, eps, STAT_BLOCK_P, REDUCE_BLOCK_C),
    )

    apply_kernel = _apply_kernel.with_hints(occupancy=apply_occupancy)
    ct.launch(
        stream,
        (num_m_apply, num_c_apply, 1),
        apply_kernel,
        (input, gamma, beta, stats, output, APPLY_BLOCK_M, APPLY_BLOCK_C),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "REDUCE_BLOCK_N": REDUCE_BLOCK_N,
        "REDUCE_BLOCK_C": REDUCE_BLOCK_C,
        "STAT_BLOCK_P": STAT_BLOCK_P,
        "APPLY_BLOCK_M": APPLY_BLOCK_M,
        "APPLY_BLOCK_C": APPLY_BLOCK_C,
        "partial_occupancy": partial_occupancy,
        "stats_occupancy": stats_occupancy,
        "apply_occupancy": apply_occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
