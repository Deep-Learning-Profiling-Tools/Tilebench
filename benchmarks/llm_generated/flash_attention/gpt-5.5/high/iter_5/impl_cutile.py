from types import SimpleNamespace
import math
import torch
import cuda.tile as ct
import numpy as np
from cuda.tile import RoundingMode as RMd

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

_last_autotune_config: dict = {}


@ct.kernel
def _flash_attention_kernel(
    q,
    k,
    v,
    out,
    scale_log2: float,
    N: ConstInt,
    D: ConstInt,
    CAUSAL: ConstBool,
    BM: ConstInt,
    BN: ConstInt,
    BD: ConstInt,
):
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)

    offs_m = (bid_m * BM + ct.arange(BM, dtype=np.int32))[:, None]
    offs_n_base = ct.arange(BN, dtype=np.int32)[None, :]

    q_tile = ct.load(
        q,
        index=(bid_bh, bid_m, 0),
        shape=(1, BM, BD),
    ).reshape((BM, BD))

    m_i = ct.full((BM, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((BM, 1), 0.0, dtype=np.float32)
    acc = ct.full((BM, BD), 0.0, dtype=np.float32)

    if CAUSAL:
        for j in range(0, ct.cdiv(N, BN)):
            if (j + 1) * BN <= bid_m * BM:
                k_tile = ct.load(
                    k,
                    index=(bid_bh, 0, j),
                    shape=(1, BD, BN),
                    order=(0, 2, 1),
                ).reshape((BD, BN))

                qk = ct.mma(
                    q_tile,
                    k_tile,
                    ct.full((BM, BN), 0.0, dtype=np.float32),
                )
                qk = qk * scale_log2

                m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
                p = ct.exp2(qk - m_ij, flush_to_zero=True)
                alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

                l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
                acc = acc * alpha
                m_i = m_ij

                v_tile = ct.load(
                    v,
                    index=(bid_bh, j, 0),
                    shape=(1, BN, BD),
                ).reshape((BN, BD))

                acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
            elif j * BN <= bid_m * BM + (BM - 1):
                offs_n = j * BN + offs_n_base

                k_tile = ct.load(
                    k,
                    index=(bid_bh, 0, j),
                    shape=(1, BD, BN),
                    order=(0, 2, 1),
                ).reshape((BD, BN))

                qk = ct.mma(
                    q_tile,
                    k_tile,
                    ct.full((BM, BN), 0.0, dtype=np.float32),
                )
                qk = qk * scale_log2
                qk = ct.where(offs_m >= offs_n, qk, -np.inf)

                m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
                p = ct.exp2(qk - m_ij, flush_to_zero=True)
                alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

                l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
                acc = acc * alpha
                m_i = m_ij

                v_tile = ct.load(
                    v,
                    index=(bid_bh, j, 0),
                    shape=(1, BN, BD),
                ).reshape((BN, BD))

                acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
    else:
        for j in range(0, ct.cdiv(N, BN)):
            k_tile = ct.load(
                k,
                index=(bid_bh, 0, j),
                shape=(1, BD, BN),
                order=(0, 2, 1),
            ).reshape((BD, BN))

            qk = ct.mma(
                q_tile,
                k_tile,
                ct.full((BM, BN), 0.0, dtype=np.float32),
            )
            qk = qk * scale_log2

            m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
            p = ct.exp2(qk - m_ij, flush_to_zero=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

            l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
            acc = acc * alpha
            m_i = m_ij

            v_tile = ct.load(
                v,
                index=(bid_bh, j, 0),
                shape=(1, BN, BD),
            ).reshape((BN, BD))

            acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)

    out_tile = ct.truediv(
        acc,
        l_i,
        flush_to_zero=True,
        rounding_mode=RMd.APPROX,
    )
    out_tile = ct.astype(out_tile, q.dtype)
    ct.store(
        out,
        index=(bid_bh, bid_m, 0),
        tile=ct.reshape(out_tile, (1, BM, BD)),
    )


_SEARCH_SPACE = [
    SimpleNamespace(tm=64, tn=128, occupancy=1),
    SimpleNamespace(tm=64, tn=128, occupancy=2),
    SimpleNamespace(tm=64, tn=128, occupancy=4),
    SimpleNamespace(tm=64, tn=256, occupancy=1),
    SimpleNamespace(tm=64, tn=256, occupancy=2),
    SimpleNamespace(tm=128, tn=64, occupancy=1),
    SimpleNamespace(tm=128, tn=64, occupancy=2),
    SimpleNamespace(tm=128, tn=64, occupancy=4),
    SimpleNamespace(tm=128, tn=128, occupancy=1),
    SimpleNamespace(tm=128, tn=128, occupancy=2),
    SimpleNamespace(tm=128, tn=128, occupancy=4),
    SimpleNamespace(tm=128, tn=256, occupancy=1),
    SimpleNamespace(tm=128, tn=256, occupancy=2),
    SimpleNamespace(tm=256, tn=64, occupancy=1),
    SimpleNamespace(tm=256, tn=64, occupancy=2),
]

_tuner = CutileAutotuner(_flash_attention_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    B, H, N, D = q.shape
    output = torch.empty_like(q)

    q3 = q.view(B * H, N, D)
    k3 = k.view(B * H, N, D)
    v3 = v.view(B * H, N, D)
    out3 = output.view(B * H, N, D)

    bd = max(32, _next_power_of_2(int(D)))
    scale_log2 = (1.0 / math.sqrt(float(D))) * 1.4426950408889634
    causal = bool(causal)

    stream = torch.cuda.current_stream()

    cfg = _tuner.tune_or_cached(
        shape_key=(B, H, N, D, causal, str(q.dtype)),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(N, cfg.tm), B * H, 1),
        args_fn=lambda cfg: (
            q3,
            k3,
            v3,
            out3,
            scale_log2,
            N,
            D,
            causal,
            cfg.tm,
            cfg.tn,
            bd,
        ),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update(
        {
            "tm": cfg.tm,
            "tn": cfg.tn,
            "bd": bd,
            "occupancy": cfg.occupancy,
        }
    )

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    ct.launch(
        stream,
        (ct.cdiv(N, cfg.tm), B * H, 1),
        kernel,
        (
            q3,
            k3,
            v3,
            out3,
            scale_log2,
            N,
            D,
            causal,
            cfg.tm,
            cfg.tn,
            bd,
        ),
    )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
