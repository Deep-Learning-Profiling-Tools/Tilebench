```python title="impl_triton.py"
import math
import torch
import triton
import triton.language as tl


_LOG2E = 1.4426950408889634


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SM_SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    base = pid_bh * N_CTX * HEAD_DIM

    q = tl.load(
        q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )

    qk_scale = SM_SCALE * _LOG2E

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.full((BLOCK_M, BLOCK_D), 0.0, tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n

        k_t = tl.load(
            k_ptr + base + cols[None, :] * HEAD_DIM + offs_d[:, None],
            mask=(cols[None, :] < N_CTX) & (offs_d[:, None] < HEAD_DIM),
            other=0.0,
        )

        qk = tl.dot(q, k_t, out_dtype=tl.float32) * qk_scale

        if CAUSAL:
            qk = tl.where(
                (cols[None, :] <= offs_m[:, None]) & (cols[None, :] < N_CTX),
                qk,
                -float("inf"),
            )
        else:
            qk = tl.where(cols[None, :] < N_CTX, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)

        v = tl.load(
            v_ptr + base + cols[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(cols[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )

        acc = acc * alpha[:, None] + tl.dot(
            p.to(tl.float16),
            v,
            out_dtype=tl.float32,
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out = acc / l_i[:, None]

    tl.store(
        o_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
        out,
        mask=(offs_m[:, None] < N_CTX) & (offs_d[None, :] < HEAD_DIM),
    )


_flash_attention_fwd_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    ],
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
)(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    batch_size, n_heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q)

    batch_heads = batch_size * n_heads
    block_d = max(32, _next_power_of_2(head_dim))
    sm_scale = 1.0 / math.sqrt(head_dim)

    grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), batch_heads)

    _flash_attention_fwd_kernel_autotuned[grid](
        q,
        k,
        v,
        output,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        SM_SCALE=sm_scale,
        CAUSAL=bool(causal),
        BLOCK_D=block_d,
    )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_flash_attention_fwd_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M": cfg.kwargs["BLOCK_M"],
        "BLOCK_N": cfg.kwargs["BLOCK_N"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
```

```python title="impl_cutile.py"
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


_tuner = CutileAutotuner(_flash_attention_fwd_kernel)


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

    bd = max(32, _next_power_of_2(head_dim))
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

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
```
