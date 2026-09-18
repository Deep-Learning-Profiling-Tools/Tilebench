Standard FlashAttention forward: each program handles a `BLOCK_M` tile of queries and streams `K/V` in `BLOCK_N` chunks while maintaining `(m_i, l_i, acc)` online-softmax state. Triton uses block pointers with the standard FA pattern; cuTile uses `ct.mma` with transposed K loads via `order=(0,1,3,2)` and an unrolled K/V tile loop (N_CTX made `ConstInt`).

```python title="impl_triton.py"
import math
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
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
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504089

    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k) * qk_scale

        if CAUSAL:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0.0, -1.0e6)

        # Guard tail in N (no-op when N_CTX divisible by BLOCK_N)
        n_oob = (start_n + offs_n) >= N_CTX
        qk = qk + tl.where(n_oob[None, :], -1.0e6, 0.0)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        acc = acc * alpha[:, None]
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(Out.dtype.element_ty), boundary_check=(0,))


def run(q, k, v, causal=True, **kwargs):
    Z, H, N_CTX, HEAD_DIM = q.shape
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    BLOCK_M = 128
    BLOCK_N = 64
    num_warps = 8
    num_stages = 3

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


@ct.kernel
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
    l_i = ct.full((M, 1), 0.0, dtype=np.float32)
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
        qk = qk * qk_scale

        if CAUSAL:
            offs_n = (j * N + ct.arange(N, dtype=np.int32))[None, :]
            mask = offs_m >= offs_n
            qk = ct.where(mask, qk, -1.0e6)

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
    occupancy = 2

    grid = (ct.cdiv(N_CTX, M), Z * H, 1)
    kernel = _fmha_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        torch.cuda.current_stream(), grid, kernel,
        (q, k, v, out, float(sm_scale), int(H), int(N_CTX),
         int(D), M, N, bool(causal)),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"M": M, "N": N, "occupancy": occupancy})
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
