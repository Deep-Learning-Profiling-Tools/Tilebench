import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _flash_attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    sm_scale,
    N_HEADS: tl.constexpr,
    N_CTX,
    HEAD_DIM,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    EVEN_Q: tl.constexpr,
    EVEN_KV: tl.constexpr,
    EVEN_D: tl.constexpr,
    PRESCALE_Q: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_b = pid_bh // N_HEADS
    off_h = pid_bh - off_b * N_HEADS

    tl.assume(stride_qd == 1)
    tl.assume(stride_kd == 1)
    tl.assume(stride_vd == 1)
    tl.assume(stride_od == 1)

    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    k_base = k_ptr + off_b * stride_kb + off_h * stride_kh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    o_base = o_ptr + off_b * stride_ob + off_h * stride_oh

    q_block = tl.make_block_ptr(
        base=q_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qs, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )

    k_block = tl.make_block_ptr(
        base=k_base,
        shape=(HEAD_DIM, N_CTX),
        strides=(stride_kd, stride_ks),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )

    v_block = tl.make_block_ptr(
        base=v_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_vs, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    if EVEN_Q:
        q = tl.load(q_block)
    else:
        q = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")

    qk_scale = sm_scale * 1.4426950408889634
    if PRESCALE_Q:
        q = (q * qk_scale).to(tl.float16)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if IS_CAUSAL:
        row_start = pid_m * BLOCK_M
        row_end = tl.minimum(N_CTX, (pid_m + 1) * BLOCK_M)

        full_end = ((row_start + 1) // BLOCK_N) * BLOCK_N
        full_end = tl.minimum(full_end, N_CTX)

        for start_n in tl.range(
            0, full_end, BLOCK_N,
            num_stages=NUM_STAGES,
            warp_specialize=WARP_SPECIALIZE,
        ):
            if EVEN_D:
                k = tl.load(k_block)
                v = tl.load(v_block)
            else:
                k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
                v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k)
            if not PRESCALE_Q:
                qk = qk * qk_scale

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            k_block = tl.advance(k_block, (0, BLOCK_N))
            v_block = tl.advance(v_block, (BLOCK_N, 0))

        for start_n in tl.range(
            full_end, row_end, BLOCK_N,
            num_stages=NUM_STAGES,
            warp_specialize=WARP_SPECIALIZE,
        ):
            if EVEN_KV:
                k = tl.load(k_block)
                v = tl.load(v_block)
            else:
                k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
                v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k)
            if not PRESCALE_Q:
                qk = qk * qk_scale

            cols = start_n + offs_n
            if EVEN_KV:
                valid = offs_m[:, None] >= cols[None, :]
            else:
                valid = (cols[None, :] < N_CTX) & (offs_m[:, None] >= cols[None, :])
            qk = tl.where(valid, qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            k_block = tl.advance(k_block, (0, BLOCK_N))
            v_block = tl.advance(v_block, (BLOCK_N, 0))
    else:
        for start_n in tl.range(
            0, N_CTX, BLOCK_N,
            num_stages=NUM_STAGES,
            warp_specialize=WARP_SPECIALIZE,
        ):
            if EVEN_KV:
                k = tl.load(k_block)
                v = tl.load(v_block)
            else:
                k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
                v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")

            qk = tl.dot(q, k)
            if not PRESCALE_Q:
                qk = qk * qk_scale

            if not EVEN_KV:
                cols = start_n + offs_n
                qk = tl.where(cols[None, :] < N_CTX, qk, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.math.exp2(m_i - m_new)
            p = tl.math.exp2(qk - m_new[:, None])

            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), v, acc)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            m_i = m_new

            k_block = tl.advance(k_block, (0, BLOCK_N))
            v_block = tl.advance(v_block, (BLOCK_N, 0))

    acc = acc / l_i[:, None]

    o_block = tl.make_block_ptr(
        base=o_base,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_os, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    if EVEN_Q:
        tl.store(o_block, acc.to(tl.float16))
    else:
        tl.store(o_block, acc.to(tl.float16), boundary_check=(0, 1))


def run(q, k, v, causal=True, **kwargs):
    output = torch.empty_like(q)

    batch_size = q.shape[0]
    n_heads = q.shape[1]
    seq_len = q.shape[2]
    head_dim = q.shape[3]

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = 128
    num_warps = 8
    num_stages = 3
    warp_specialize = True
    prescale_q = True

    even_q = (seq_len % BLOCK_M == 0) and (head_dim == BLOCK_D)
    even_kv = (seq_len % BLOCK_N == 0) and (head_dim == BLOCK_D)
    even_d = head_dim == BLOCK_D

    sm_scale = head_dim ** -0.5
    grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * n_heads)

    _flash_attention_fwd_kernel[grid](
        q, k, v, output,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        sm_scale,
        N_HEADS=n_heads,
        N_CTX=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=bool(causal),
        NUM_STAGES=num_stages,
        WARP_SPECIALIZE=warp_specialize,
        EVEN_Q=even_q,
        EVEN_KV=even_kv,
        EVEN_D=even_d,
        PRESCALE_Q=prescale_q,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_D": BLOCK_D,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "warp_specialize": warp_specialize,
        "causal": bool(causal),
        "EVEN_Q": even_q,
        "EVEN_KV": even_kv,
        "PRESCALE_Q": prescale_q,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
