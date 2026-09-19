import math
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _fmha_kernel(Q, K, V, Out, sm_scale: float,
                 H: ConstInt, N_CTX: ConstInt,
                 D: ConstInt, M: ConstInt, N: ConstInt,
                 CAUSAL: ConstBool):
    start_m = ct.bid(0)
    off_bh = ct.bid(1)
    batch = off_bh // H
    head = off_bh % H

    qk_scale = sm_scale * (1.0 / math.log(2.0))

    m_i = ct.full((M, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((M, 1), 1.0, dtype=np.float32)
    acc = ct.full((M, D), 0.0, dtype=np.float32)

    q = ct.load(Q, index=(batch, head, start_m, 0),
                shape=(1, 1, M, D),
                padding_mode=ct.PaddingMode.ZERO).reshape((M, D))

    num_n_tiles = ct.cdiv(N_CTX, N)
    offs_m = (start_m * M + ct.arange(M, dtype=np.int32))[:, None]

    for j in range(0, num_n_tiles):
        k = ct.load(K, index=(batch, head, 0, j),
                    shape=(1, 1, D, N), order=(0, 1, 3, 2),
                    padding_mode=ct.PaddingMode.ZERO).reshape((D, N))
        qk = ct.mma(q, k, ct.full((M, N), 0.0, dtype=np.float32))

        if CAUSAL:
            offs_n = (j * N + ct.arange(N, dtype=np.int32))[None, :]
            mask = offs_m >= offs_n
            qk = qk * qk_scale + ct.where(mask, 0.0, -1.0e6)
        else:
            qk = qk * qk_scale

        m_ij = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True))
        p = ct.exp2(qk - m_ij, flush_to_zero=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)
        l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha

        v = ct.load(V, index=(batch, head, j, 0),
                    shape=(1, 1, N, D),
                    padding_mode=ct.PaddingMode.ZERO).reshape((N, D))
        acc = ct.mma(ct.astype(p, Q.dtype), v, acc)
        m_i = m_ij

    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    ct.store(Out, index=(batch, head, start_m, 0),
             tile=ct.astype(acc, Out.dtype).reshape((1, 1, M, D)))


def run(q, k, v, causal=True, **kwargs):
    Z, H, N_CTX, D = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)

    M = 128
    N = 64

    grid = (ct.cdiv(N_CTX, M), Z * H, 1)
    ct.launch(
        torch.cuda.current_stream(), grid, _fmha_kernel,
        (q, k, v, out, float(sm_scale), int(H), int(N_CTX),
         int(D), M, N, bool(causal)),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"M": M, "N": N, "occupancy": 2})
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
