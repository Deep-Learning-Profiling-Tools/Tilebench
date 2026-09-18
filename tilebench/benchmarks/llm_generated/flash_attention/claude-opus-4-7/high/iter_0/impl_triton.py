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
