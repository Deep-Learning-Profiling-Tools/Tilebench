import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _block_sparse_attention_local3_kernel(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    O,
    csr_row_stride_h,
    csr_col_stride_h,
    num_layout,
    softmax_scale,
    num_heads,
    num_kv_heads,
    total_seq_len,
    head_dim,
    q_stride_b,
    q_stride_h,
    q_stride_m,
    q_stride_d,
    k_stride_b,
    k_stride_h,
    k_stride_m,
    k_stride_d,
    v_stride_b,
    v_stride_h,
    v_stride_m,
    v_stride_d,
    o_stride_b,
    o_stride_h,
    o_stride_m,
    o_stride_d,
    BLOCK_M_T: tl.constexpr,
    BLOCK_N_T: tl.constexpr,
    BLOCK_D_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    b = pid_bh // num_heads
    h = pid_bh - b * num_heads
    heads_per_kv = num_heads // num_kv_heads
    h_kv = h // heads_per_kv

    offs_m_local = tl.arange(0, BLOCK_M_T)
    offs_m = pid_m * BLOCK_M_T + offs_m_local
    offs_n = tl.arange(0, BLOCK_N_T)
    offs_d = tl.arange(0, BLOCK_D_T)

    q_ptrs = (
        Q
        + b * q_stride_b
        + h * q_stride_h
        + offs_m[:, None] * q_stride_m
        + offs_d[None, :] * q_stride_d
    )
    q = tl.load(q_ptrs, eviction_policy="evict_first")

    m_i = tl.full((BLOCK_M_T,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_T, BLOCK_D_T), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634
    causal_mask = offs_m_local[:, None] >= offs_n[None, :]

    for t in tl.static_range(0, 3):
        raw_col_block = pid_m + t - 2
        active = raw_col_block >= 0
        col_block = tl.maximum(raw_col_block, 0)
        offs_cols = col_block * BLOCK_N_T + offs_n

        k_ptrs = (
            K
            + b * k_stride_b
            + h_kv * k_stride_h
            + offs_d[:, None] * k_stride_d
            + offs_cols[None, :] * k_stride_m
        )
        k = tl.load(k_ptrs, eviction_policy="evict_last")

        qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale

        if t == 2:
            qk = tl.where(causal_mask, qk, -float("inf"))
        else:
            qk = tl.where(active, qk, -float("inf"))

        m_block = tl.max(qk, axis=1)
        m_cand = tl.maximum(m_i, m_block)
        m_new = tl.where(active, m_cand, m_i)

        alpha_raw = tl.math.exp2(m_i - m_new)
        alpha = tl.where(active, alpha_raw, 1.0)

        p_raw = tl.math.exp2(qk - m_new[:, None])
        p = tl.where(active, p_raw, 0.0).to(tl.float16)

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
            V
            + b * v_stride_b
            + h_kv * v_stride_h
            + offs_cols[:, None] * v_stride_m
            + offs_d[None, :] * v_stride_d
        )
        v = tl.load(v_ptrs, eviction_policy="evict_last")

        acc = tl.dot(p, v, acc, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = (
        O
        + b * o_stride_b
        + h * o_stride_h
        + offs_m[:, None] * o_stride_m
        + offs_d[None, :] * o_stride_d
    )
    tl.store(o_ptrs, acc)


def run(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    layout_csr_row_stride_h,
    layout_csr_col_stride_h,
    num_layout,
    softmax_scale,
    num_heads,
    num_kv_heads,
    total_seq_len,
    BLOCK_M,
    EVEN_M,
    BLOCK_N,
    EVEN_N,
    BLOCK_D,
    NUM_D_BLOCKS,
):
    out = torch.empty_like(Q)

    bm = int(BLOCK_M)
    bn = int(BLOCK_N)
    bd = int(BLOCK_D)

    num_warps = 4
    num_stages = 3

    grid = (Q.shape[0] * int(num_heads), triton.cdiv(int(total_seq_len), bm))

    _block_sparse_attention_local3_kernel[grid](
        Q,
        K,
        V,
        layout_csr_row_indices,
        layout_csr_col_indices,
        out,
        int(layout_csr_row_stride_h),
        int(layout_csr_col_stride_h),
        int(num_layout),
        float(softmax_scale),
        int(num_heads),
        int(num_kv_heads),
        int(total_seq_len),
        int(Q.shape[3]),
        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),
        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),
        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_M_T=bm,
        BLOCK_N_T=bn,
        BLOCK_D_T=bd,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_D": bd,
            "WINDOW_BLOCKS": 2,
            "P_FP16_SUM": 1,
            "LOCAL_CAUSAL_MASK": 1,
            "grid_bh_major": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
