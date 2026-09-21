from types import SimpleNamespace

import torch
import cuda.tile as ct
import math
from cuda.tile import RoundingMode as RMd

from tilebench.core.cutile_autotune import CutileAutotuner

INV_LOG_2 = 1.0 / math.log(2)
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_last_autotune_config: dict = {}

_DEFAULT_CONFIG = SimpleNamespace(tile_m=64, tile_n=64, occupancy=16)
_SEARCH_SPACE = [
    SimpleNamespace(tile_m=tm, tile_n=tn, occupancy=occ)
    for tm in [64, 128]
    for tn in [32, 64, 128]
    for occ in [4, 8, 16, 32]
]

@ct.kernel
def fwd_kernel(Q, K, V, Out,
                qk_scale: float,
                input_pos: int,
                TILE_D: ConstInt,
                H: ConstInt,
                TILE_M: ConstInt,
                TILE_N: ConstInt,
                QUERY_GROUP_SIZE: ConstInt,
                CAUSAL: ConstBool,
                EVEN_K: ConstBool):

    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    batch_idx = bid_y // H
    head_idx = bid_y % H
    off_kv_h = head_idx // QUERY_GROUP_SIZE


    qk_scale = qk_scale * INV_LOG_2


    offs_m = bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    offs_m += input_pos
    offs_m = offs_m[:, None]


    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)
    offs_n_tile = offs_n_tile[None, :]


    m_i = ct.full((TILE_M, 1), -float('inf'), dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)


    q = ct.load(
        Q, index=(batch_idx, head_idx, bid_x, 0), shape=(1, 1, TILE_M, TILE_D)
    ).reshape((TILE_M, TILE_D))


    m_end = input_pos + (bid_x + 1) * TILE_M
    k_seqlen = K.shape[2]
    if CAUSAL:

        mask_start = (input_pos + bid_x * TILE_M) // TILE_N

        mask_start = min(mask_start, k_seqlen // TILE_N)
        Tc = ct.cdiv(min(m_end, k_seqlen), TILE_N)
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N


    for j in range(0, Tc):

        k = ct.load(
            K, index=(batch_idx, off_kv_h, 0, j), shape=(1, 1, TILE_D, TILE_N),
            order=(0, 1, 3, 2),
            latency=2,
        )
        k = k.reshape((TILE_D, TILE_N))
        qk = ct.full((TILE_M, TILE_N), 0., dtype=ct.float32)
        qk = ct.mma(q, k, qk)


        if (CAUSAL or not EVEN_K) and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            mask = ct.full((TILE_M, TILE_N), True, dtype=ct.bool_)

            if not EVEN_K:
                mask = mask & (offs_n < k_seqlen)

            if CAUSAL:
                mask = mask & (offs_m >= offs_n)
            mask = ct.where(mask, 0.0, -float('inf'))
            qk += mask


        m_ij = max(m_i, ct.max(qk, axis=-1, keepdims=True) * qk_scale)
        qk = qk * qk_scale - m_ij


        p = ct.exp2(qk, flush_to_zero=True)
        l_ij = ct.sum(p, axis=-1, keepdims=True)
        alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

        l_i = l_i * alpha + l_ij

        acc = acc * alpha


        v = ct.load(
            V, index=(batch_idx, off_kv_h, j, 0), shape=(1, 1, TILE_N, TILE_D),
            latency=4,
        ).reshape((TILE_N, TILE_D))
        p = p.astype(Q.dtype)
        acc = ct.mma(p, v, acc)
        m_i = m_ij


    acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
    acc = acc.reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)


_tuner = CutileAutotuner(fwd_kernel)


def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True, autotune: bool = False, **kwargs):

    Batch, Heads, SeqLen_Q, D_k = q.shape

    input_pos = 0
    qk_scale = 1.0 / math.sqrt(D_k)
    query_group_size = 1

    Out = torch.empty_like(q)
    stream = torch.cuda.current_stream()

    def build_args(tile_m, tile_n):
        return (
            q, k, v, Out,
            qk_scale,
            input_pos,
            D_k,
            Heads,
            tile_m,
            tile_n,
            query_group_size,
            causal,
            (SeqLen_Q % tile_n) == 0,
        )

    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(SeqLen_Q, D_k, Heads, causal, str(q.dtype)),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: (math.ceil(SeqLen_Q / cfg.tile_m), Batch * Heads, 1),
            args_fn=lambda cfg: build_args(cfg.tile_m, cfg.tile_n),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({
            "tile_m": cfg.tile_m,
            "tile_n": cfg.tile_n,
            "occupancy": cfg.occupancy,
        })
    else:
        cfg = _DEFAULT_CONFIG

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    grid = (math.ceil(SeqLen_Q / cfg.tile_m), Batch * Heads, 1)
    ct.launch(stream, grid, kernel, build_args(cfg.tile_m, cfg.tile_n))

    return Out

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
