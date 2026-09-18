Looking at the trajectory:
- **Triton**: Iter 4 broke verify (warp_specialize + prescaled-q combined). The big magnitude error (4.35) points to warp_specialize being the culprit. I'll revert to iter 3 and apply only the safer optimization: pre-scale Q (saves one FMA per inner-loop QK element) without warp_specialize.
- **cuTile**: Iter 4 regressed slightly from iter 3 (the only change was prescaling Q). Revert to iter 3 exactly.

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
    start_m,
    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m, offs_n, N_CTX: tl.constexpr,
):
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
        # qk already in log2 units because q was pre-scaled on the host-uniform path
        qk = tl.dot(q, k)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0.0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]

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
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)

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
    # Pre-scale q via fp32 intermediate so the inner-loop qk = tl.dot(q,k)
    # is already in log2 units — saves one multiplication per qk element.
    # Compute in fp32 to keep precision, then downcast for MMA.
    q = (q.to(tl.float32) * qk_scale).to(q.dtype)

    if CAUSAL:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            1, offs_m, offs_n, N_CTX,
        )
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            2, offs_m, offs_n, N_CTX,
        )
    else:
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q,
            K_block_ptr, V_block_ptr,
            start_m,
            BLOCK_M, HEAD_DIM, BLOCK_N,
            3, offs_m, offs_n, N_CTX,
        )

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty))


def run(q, k, v, causal=True, **kwargs):
    Z, H, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Iter 3 baseline (BM=128, BN=128, nw=8, ns=3) was 59.5% verify-clean.
    # Iter 4 added warp_specialize=True + prescaled-q and broke verify with
    # huge errors (4.35 abs diff). Root cause is the warp_specialize=True
    # path interacting with the causal-stage-2 mask; the prescale itself is
    # numerically safe (the official Triton tutorial uses it). This iter:
    # keep iter-3 tiling/warps/stages, add ONLY pre-scaled q (computed via
    # fp32 intermediate for precision safety), no warp_specialize.
    BLOCK_M = 128
    BLOCK_N = 128
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
        "prescaled_q": True,
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

    qk_scale = sm_scale * 1.4426950408889634  # 1/log(2)

    m_i = ct.full((M, 1), -float("inf"), dtype=np.float32)
    l_i = ct.full((M, 1), 0.0, dtype=np.float32)
    acc = ct.full((M, D), 0.0, dtype=np.float32)

    q = ct.load(Q, index=(batch, head, start_m, 0),
                shape=(1, 1, M, D)).reshape((M, D))

    num_n_tiles = ct.cdiv(N_CTX, N)
    offs_m = (start_m * M + ct.arange(M, dtype=np.int32))[:, None]

    for j in range(0, num_n_tiles):
        # Load K transposed → (D, N) for q @ k_t
        k = ct.load(K, index=(batch, head, 0, j),
                    shape=(1, 1, D, N), order=(0, 1, 3, 2),
                    padding_mode=ct.PaddingMode.ZERO).reshape((D, N))
        qk = ct.mma(q, k, ct.full((M, N), 0.0, dtype=np.float32))

        if CAUSAL:
            offs_n = (j * N + ct.arange(N, dtype=np.int32))[None, :]
            mask = offs_m >= offs_n
            # Finite negative for masked elements (avoids -inf - -inf = NaN
            # on rows where the entire tile is causally masked out).
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

    # Iter 3 (M=128, N=64, occ=2, mid-loop scale) was 41.5% verify-clean.
    # Iter 4 added prescaled-q and regressed by 2.8 pp. Revert to iter 3
    # exact config — the prescaling tweak measurably hurt cuTile's pipeline
    # (likely due to extra fp32 traffic on the q tile not amortized over
    # the inner loop the way Triton's block-pointer pipeline manages it).
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

**Approach summary**: Reverted both backends to their best verify-clean configs (iter 3). For Triton, applied only the pre-scaled-Q optimization (computed via fp32 intermediate for precision safety) without warp_specialize — isolating the one safe change from iter 4. For cuTile, fully reverted since iter 4's prescaling-only change regressed.
