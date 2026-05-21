from types import SimpleNamespace
import math
import torch
import cuda.tile as ct
from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]
ConstFloat = ct.Constant[float]

_last_autotune_config: dict = {}

_SEARCH_SPACE = [
    SimpleNamespace(bm=64, bn=64, occupancy=4),
    SimpleNamespace(bm=64, bn=64, occupancy=8),
    SimpleNamespace(bm=64, bn=64, occupancy=16),
    SimpleNamespace(bm=64, bn=128, occupancy=4),
    SimpleNamespace(bm=64, bn=128, occupancy=8),
    SimpleNamespace(bm=64, bn=128, occupancy=16),
    SimpleNamespace(bm=128, bn=64, occupancy=4),
    SimpleNamespace(bm=128, bn=64, occupancy=8),
]


@ct.kernel
def _flash_attention_fwd_kernel(
    q,
    k,
    v,
    out,
    N_CTX: ConstInt,
    HEAD_DIM: ConstInt,
    SM_SCALE: ConstFloat,
    CAUSAL: ConstBool,
    BM: ConstInt,
    BN: ConstInt,
    BD: ConstInt,
):
    pid_m = ct.bid(0)
    pid_bh = ct.bid(1)

    q_tile_id = pid_bh * (N_CTX // BM) + pid_m

    q_tile = ct.load(
        q,
        index=(q_tile_id, 0),
        shape=(BM, BD),
        padding_mode=ct.PaddingMode.ZERO,
    )

    acc = ct.full((BM, BD), 0.0, ct.float32)
    m_i = ct.full((BM,), -float("inf"), ct.float32)
    l_i = ct.full((BM,), 0.0, ct.float32)

    rows = pid_m * BM + ct.arange(BM, dtype=ct.int32)
    qk_scale = SM_SCALE * 1.4426950408889634

    if CAUSAL:
        for start_n in range(0, N_CTX, BN):
            if start_n < (pid_m + 1) * BM:
                kv_tile_id = pid_bh * (N_CTX // BN) + (start_n // BN)

                k_tile = ct.load(
                    k,
                    index=(kv_tile_id, 0),
                    shape=(BN, BD),
                    padding_mode=ct.PaddingMode.ZERO,
                )

                scores = ct.mma(
                    q_tile,
                    ct.transpose(k_tile),
                    ct.full((BM, BN), 0.0, ct.float32),
                )
                scores = scores * qk_scale

                cols = start_n + ct.arange(BN, dtype=ct.int32)
                scores = ct.where(cols[None, :] <= rows[:, None], scores, -float("inf"))

                m_ij = ct.max(scores, axis=1)
                m_new = ct.maximum(m_i, m_ij)

                p = ct.exp2(scores - m_new[:, None])
                alpha = ct.exp2(m_i - m_new)

                v_tile = ct.load(
                    v,
                    index=(kv_tile_id, 0),
                    shape=(BN, BD),
                    padding_mode=ct.PaddingMode.ZERO,
                )

                acc = acc * alpha[:, None]
                acc = ct.mma(
                    p.to(ct.float16),
                    v_tile,
                    acc,
                )
                l_i = l_i * alpha + ct.sum(p, axis=1)
                m_i = m_new
    else:
        for start_n in range(0, N_CTX, BN):
            kv_tile_id = pid_bh * (N_CTX // BN) + (start_n // BN)

            k_tile = ct.load(
                k,
                index=(kv_tile_id, 0),
                shape=(BN, BD),
                padding_mode=ct.PaddingMode.ZERO,
            )

            scores = ct.mma(
                q_tile,
                ct.transpose(k_tile),
                ct.full((BM, BN), 0.0, ct.float32),
            )
            scores = scores * qk_scale

            m_ij = ct.max(scores, axis=1)
            m_new = ct.maximum(m_i, m_ij)

            p = ct.exp2(scores - m_new[:, None])
            alpha = ct.exp2(m_i - m_new)

            v_tile = ct.load(
                v,
                index=(kv_tile_id, 0),
                shape=(BN, BD),
                padding_mode=ct.PaddingMode.ZERO,
            )

            acc = acc * alpha[:, None]
            acc = ct.mma(
                p.to(ct.float16),
                v_tile,
                acc,
            )
            l_i = l_i * alpha + ct.sum(p, axis=1)
            m_i = m_new

    out_tile = acc / l_i[:, None]

    ct.store(
        out,
        index=(q_tile_id, 0),
        tile=out_tile,
    )


@ct.kernel
def _flash_attention_early_fix_kernel(
    q,
    k,
    v,
    out,
    N_CTX: ConstInt,
    HEAD_DIM: ConstInt,
    SM_SCALE: ConstFloat,
    EARLY_M: ConstInt,
    BD: ConstInt,
):
    pid_bh = ct.bid(0)
    tile_id = pid_bh * (N_CTX // EARLY_M)

    q_tile = ct.load(
        q,
        index=(tile_id, 0),
        shape=(EARLY_M, BD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    k_tile = ct.load(
        k,
        index=(tile_id, 0),
        shape=(EARLY_M, BD),
        padding_mode=ct.PaddingMode.ZERO,
    )

    scores = ct.mma(
        q_tile,
        ct.transpose(k_tile),
        ct.full((EARLY_M, EARLY_M), 0.0, ct.float32),
    )
    scores = scores * (SM_SCALE * 1.4426950408889634)

    rows = ct.arange(EARLY_M, dtype=ct.int32)
    cols = ct.arange(EARLY_M, dtype=ct.int32)
    scores = ct.where(cols[None, :] <= rows[:, None], scores, -float("inf"))

    m = ct.max(scores, axis=1)
    p = ct.exp2(scores - m[:, None])
    l = ct.sum(p, axis=1)
    p = p / l[:, None]

    v_tile = ct.load(
        v,
        index=(tile_id, 0),
        shape=(EARLY_M, BD),
        padding_mode=ct.PaddingMode.ZERO,
    )

    acc = ct.mma(
        p.to(ct.float16),
        v_tile,
        ct.full((EARLY_M, BD), 0.0, ct.float32),
    )

    ct.store(
        out,
        index=(tile_id, 0),
        tile=acc,
    )


_tuner = CutileAutotuner(_flash_attention_fwd_kernel)
_early_tuner = CutileAutotuner(_flash_attention_early_fix_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch_size, n_heads, seq_len, head_dim = q.shape
    batch_heads = batch_size * n_heads

    output = torch.empty_like(q)

    q2 = q.reshape(batch_heads * seq_len, head_dim)
    k2 = k.reshape(batch_heads * seq_len, head_dim)
    v2 = v.reshape(batch_heads * seq_len, head_dim)
    out2 = output.reshape(batch_heads * seq_len, head_dim)

    bd = _next_power_of_2(head_dim)
    sm_scale = 1.0 / math.sqrt(head_dim)

    stream = torch.cuda.current_stream()

    cfg = _tuner.tune_or_cached(
        shape_key=(batch_heads, seq_len, head_dim, bool(causal)),
        search_space=_SEARCH_SPACE,
        stream=stream,
        grid_fn=lambda cfg: (ct.cdiv(seq_len, cfg.bm), batch_heads, 1),
        args_fn=lambda cfg: (
            q2,
            k2,
            v2,
            out2,
            seq_len,
            head_dim,
            sm_scale,
            bool(causal),
            cfg.bm,
            cfg.bn,
            bd,
        ),
        hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
    )

    _last_autotune_config.clear()
    _last_autotune_config.update(
        {
            "bm": cfg.bm,
            "bn": cfg.bn,
            "occupancy": cfg.occupancy,
        }
    )

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)

    ct.launch(
        stream,
        (ct.cdiv(seq_len, cfg.bm), batch_heads, 1),
        kernel,
        (
            q2,
            k2,
            v2,
            out2,
            seq_len,
            head_dim,
            sm_scale,
            bool(causal),
            cfg.bm,
            cfg.bn,
            bd,
        ),
    )

    if bool(causal):
        early_kernel = _early_tuner.kernel_with_hints(occupancy=4)
        ct.launch(
            stream,
            (batch_heads, 1, 1),
            early_kernel,
            (
                q2,
                k2,
                v2,
                out2,
                seq_len,
                head_dim,
                sm_scale,
                64,
                bd,
            ),
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
