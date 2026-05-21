```python title="impl_triton.py"
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    scale_log2,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_km,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vm,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_om,
    stride_od,
    B: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    # Pre-scale Q into log2 space once per query tile.  This avoids a large
    # elementwise scale of every QK tile and keeps both matmuls on fp16 TC.
    q = (q * scale_log2).to(tl.float16)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if CAUSAL:
        full_end = (pid_m * BLOCK_M // BLOCK_N) * BLOCK_N

        for start_n in tl.range(0, full_end, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_km
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_CTX)

        for start_n in tl.range(full_end, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_km
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)
            qk = tl.where(offs_n[None, :] <= offs_m[:, None], qk, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij
    else:
        for start_n in tl.range(0, N_CTX, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)

            k = tl.load(
                k_ptr
                + b * stride_kb
                + h * stride_kh
                + offs_d[:, None] * stride_kd
                + offs_n[None, :] * stride_km
            )

            qk = tl.dot(q, k, out_dtype=tl.float32)

            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            alpha = tl.exp2(m_i - m_ij)

            v = tl.load(
                v_ptr
                + b * stride_vb
                + h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd
            )

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_ij

    out = acc / l_i[:, None]

    tl.store(
        o_ptr
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od,
        out,
    )


_flash_attention_fwd_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    ],
    key=["B", "H", "N_CTX", "D_HEAD", "CAUSAL"],
)(_flash_attention_fwd_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    B, H, N_CTX, D_HEAD = q.shape
    output = torch.empty_like(q)

    block_d = max(32, _next_power_of_2(int(D_HEAD)))
    scale_log2 = (1.0 / math.sqrt(float(D_HEAD))) * 1.4426950408889634
    causal = bool(causal)

    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]), B * H)

    _flash_attention_fwd_autotuned[grid](
        q,
        k,
        v,
        output,
        scale_log2,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        B=B,
        H=H,
        N_CTX=N_CTX,
        D_HEAD=D_HEAD,
        BLOCK_D=block_d,
        CAUSAL=causal,
    )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_flash_attention_fwd_autotuned, "best_config", None)
    if cfg is None:
        return None
    out = dict(cfg.kwargs)
    out["num_warps"] = cfg.num_warps
    out["num_stages"] = cfg.num_stages
    return out
```

```python title="impl_cutile.py"
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
    H: ConstInt,
    N: ConstInt,
    D: ConstInt,
    CAUSAL: ConstBool,
    BM: ConstInt,
    BN: ConstInt,
    BD: ConstInt,
):
    bid_m = ct.bid(0)
    bid_bh = ct.bid(1)
    b = bid_bh // H
    h = bid_bh % H

    offs_m = (bid_m * BM + ct.arange(BM, dtype=np.int32))[:, None]
    offs_n_base = ct.arange(BN, dtype=np.int32)[None, :]

    q_tile = ct.load(
        q,
        index=(b, h, bid_m, 0),
        shape=(1, 1, BM, BD),
    ).reshape((BM, BD))
    # Pre-scale Q into log2-softmax space once.  The cast keeps the QK MMA on
    # fp16 tensor cores while avoiding per-QK-tile scalar multiplication.
    q_tile = ct.astype(q_tile * scale_log2, q.dtype)

    m_i = ct.full((BM, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((BM, 1), 0.0, dtype=np.float32)
    acc = ct.full((BM, BD), 0.0, dtype=np.float32)

    if CAUSAL:
        for j in range(0, ct.cdiv(N, BN)):
            if (j + 1) * BN <= bid_m * BM:
                k_tile = ct.load(
                    k,
                    index=(b, h, 0, j),
                    shape=(1, 1, BD, BN),
                    order=(0, 1, 3, 2),
                ).reshape((BD, BN))

                qk = ct.mma(
                    q_tile,
                    k_tile,
                    ct.full((BM, BN), 0.0, dtype=np.float32),
                )

                m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
                p = ct.exp2(qk - m_ij, flush_to_zero=True)
                alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

                l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
                acc = acc * alpha

                v_tile = ct.load(
                    v,
                    index=(b, h, j, 0),
                    shape=(1, 1, BN, BD),
                ).reshape((BN, BD))

                acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
                m_i = m_ij
            elif j * BN <= bid_m * BM + (BM - 1):
                offs_n = j * BN + offs_n_base

                k_tile = ct.load(
                    k,
                    index=(b, h, 0, j),
                    shape=(1, 1, BD, BN),
                    order=(0, 1, 3, 2),
                ).reshape((BD, BN))

                qk = ct.mma(
                    q_tile,
                    k_tile,
                    ct.full((BM, BN), 0.0, dtype=np.float32),
                )
                qk = ct.where(offs_m >= offs_n, qk, -np.inf)

                m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
                p = ct.exp2(qk - m_ij, flush_to_zero=True)
                alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

                l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
                acc = acc * alpha

                v_tile = ct.load(
                    v,
                    index=(b, h, j, 0),
                    shape=(1, 1, BN, BD),
                ).reshape((BN, BD))

                acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
                m_i = m_ij
    else:
        for j in range(0, ct.cdiv(N, BN)):
            k_tile = ct.load(
                k,
                index=(b, h, 0, j),
                shape=(1, 1, BD, BN),
                order=(0, 1, 3, 2),
            ).reshape((BD, BN))

            qk = ct.mma(
                q_tile,
                k_tile,
                ct.full((BM, BN), 0.0, dtype=np.float32),
            )

            m_ij = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
            p = ct.exp2(qk - m_ij, flush_to_zero=True)
            alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)

            l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)
            acc = acc * alpha

            v_tile = ct.load(
                v,
                index=(b, h, j, 0),
                shape=(1, 1, BN, BD),
            ).reshape((BN, BD))

            acc = ct.mma(ct.astype(p, q.dtype), v_tile, acc)
            m_i = m_ij

    out_tile = ct.truediv(
        acc,
        l_i,
        flush_to_zero=True,
        rounding_mode=RMd.APPROX,
    )
    out_tile = ct.astype(out_tile, q.dtype)
    ct.store(
        out,
        index=(b, h, bid_m, 0),
        tile=ct.reshape(out_tile, (1, 1, BM, BD)),
    )


_SEARCH_SPACE = [
    SimpleNamespace(tm=64, tn=64, occupancy=1),
    SimpleNamespace(tm=64, tn=64, occupancy=2),
    SimpleNamespace(tm=64, tn=128, occupancy=1),
    SimpleNamespace(tm=64, tn=128, occupancy=2),
    SimpleNamespace(tm=64, tn=128, occupancy=4),
    SimpleNamespace(tm=128, tn=64, occupancy=1),
    SimpleNamespace(tm=128, tn=64, occupancy=2),
    SimpleNamespace(tm=128, tn=64, occupancy=4),
    SimpleNamespace(tm=128, tn=128, occupancy=1),
    SimpleNamespace(tm=128, tn=128, occupancy=2),
]

_tuner = CutileAutotuner(_flash_attention_kernel)


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def run(q, k, v, causal=True, autotune: bool = False, **kwargs):
    B, H, N, D = q.shape
    output = torch.empty_like(q)

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
            q,
            k,
            v,
            output,
            scale_log2,
            H,
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
            q,
            k,
            v,
            output,
            scale_log2,
            H,
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
```
