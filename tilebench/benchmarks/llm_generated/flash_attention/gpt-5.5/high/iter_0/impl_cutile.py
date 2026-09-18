import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel
def _flash_attention_fwd_kernel(
    q, k, v, output,
    qk_scale: float,
    SEQLEN: ConstInt,
    HEAD_DIM: ConstInt,
    N_HEADS: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
    BLOCK_D: ConstInt,
    CAUSAL: ConstBool,
):
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)

    batch = bid_bh // N_HEADS
    head = bid_bh - batch * N_HEADS

    q_tile = ct.load(
        q,
        index=(batch, head, bid_m, 0),
        shape=(1, 1, BLOCK_M, BLOCK_D),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((BLOCK_M, BLOCK_D))

    m_i = ct.full((BLOCK_M, 1), -float("inf"), dtype=np.float32)
    l_i = ct.full((BLOCK_M, 1), 0.0, dtype=np.float32)
    acc = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=np.float32)

    offs_m = bid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)[:, None]
    offs_n_base = ct.arange(BLOCK_N, dtype=np.int32)[None, :]

    n_tiles = ct.cdiv(SEQLEN, BLOCK_N)

    for j in range(0, n_tiles):
        k_tile = ct.load(
            k,
            index=(batch, head, 0, j),
            shape=(1, 1, BLOCK_D, BLOCK_N),
            order=(0, 1, 3, 2),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_D, BLOCK_N))

        qk = ct.mma(
            q_tile,
            k_tile,
            ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32),
        )
        qk = qk * qk_scale

        offs_n = j * BLOCK_N + offs_n_base
        valid = ct.broadcast_to(offs_n < SEQLEN, (BLOCK_M, BLOCK_N))
        if CAUSAL:
            valid = valid & (offs_m >= offs_n)

        qk = ct.where(valid, qk, -float("inf"))

        m_new = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True))
        p = ct.exp2(qk - m_new, flush_to_zero=True)
        alpha = ct.exp2(m_i - m_new, flush_to_zero=True)

        l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha

        v_tile = ct.load(
            v,
            index=(batch, head, j, 0),
            shape=(1, 1, BLOCK_N, BLOCK_D),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((BLOCK_N, BLOCK_D))

        acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
        m_i = m_new

    out_tile = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    out_tile = ct.astype(out_tile, q.dtype).reshape((1, 1, BLOCK_M, BLOCK_D))

    ct.store(
        output,
        index=(batch, head, bid_m, 0),
        tile=out_tile,
    )


def run(q, k, v, causal=True, **kwargs):
    output = torch.empty_like(q)

    batch_size = q.shape[0]
    n_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128
    occupancy = 2

    sm_scale_log2 = (head_dim ** -0.5) * 1.4426950408889634

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(seq_len, BLOCK_M), batch_size * n_heads, 1)
    kernel = _flash_attention_fwd_kernel.with_hints(occupancy=occupancy)

    ct.launch(
        stream,
        grid,
        kernel,
        (
            q, k, v, output,
            sm_scale_log2,
            seq_len,
            head_dim,
            n_heads,
            BLOCK_M,
            BLOCK_N,
            BLOCK_D,
            bool(causal),
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_D": BLOCK_D,
        "occupancy": occupancy,
        "causal": bool(causal),
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
