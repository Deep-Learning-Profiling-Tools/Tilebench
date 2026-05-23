Looking at the iter 0 results:
- Triton: 39.5% — the kernel does causal masking on every iteration even for blocks far from the diagonal. Splitting into a "fully attended" stage + a "diagonal masked" stage (canonical FA2 trick) should give a substantial speedup since most blocks skip the mask.
- cuTile: failed because `with_hints` doesn't exist — `occupancy` is passed to the `@ct.kernel` decorator directly.

```python title="impl_triton.py"
import math
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q,
    K_block_ptr, V_block_ptr,
    start_m, qk_scale,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m, offs_n, N_CTX: tl.constexpr,
):
    # STAGE 1: off-diagonal (no mask) — j in [0, start_m*BLOCK_M)
    # STAGE 2: diagonal (with mask) — j in [start_m*BLOCK_M, (start_m+1)*BLOCK_M)
    # STAGE 3: non-causal full range
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo = start_m * BLOCK_M
        hi = (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v = tl.load(V_block_ptr)
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc)

        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    q_base = Q + off_z * stride_qz + off_h * stride_qh
    k_base = K + off_z * stride_kz + off_h * stride_kh
    v_base = V + off_z * stride_vz + off_h * stride_vh
    o_base = Out + off_z * stride_oz + off_h * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=q_base, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=k_base, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=v_base, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=o_base, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_ok),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504089

    q = tl.load(Q_block_ptr)

    if CAUSAL:
        # Stage 1: off-diagonal (no mask)
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            1, offs_m, offs_n, N_CTX,
        )
        # Stage 2: diagonal block (masked)
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n, N_CTX,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m, qk_scale,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            3, offs_m, offs_n, N_CTX,
        )

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))


def run(q, k, v, causal=True, **kwargs):
    Z, H, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 8
    num_stages = 3

    assert N_CTX % BLOCK_M == 0, "this kernel assumes N_CTX is a multiple of BLOCK_M"

    grid = (triton.cdiv(N_CTX, BLOCK_M), Z * H, 1)
    _attn_fwd_kernel[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        Z, H, N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=bool(causal),
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "num_warps": num_warps, "num_stages": num_stages,
    })
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
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
```

The Triton version uses the canonical FA2 two-stage split (off-diagonal blocks skip the causal mask), and the cuTile version fixes the `with_hints` error by attaching `occupancy=2` directly to the `@ct.kernel` decorator.
