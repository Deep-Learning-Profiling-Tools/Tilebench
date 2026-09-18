import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _block_sparse_attention_kernel(
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
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // num_heads
    h = pid_bh - b * num_heads
    heads_per_kv = num_heads // num_kv_heads
    h_kv = h // heads_per_kv
    layout_h = h % num_layout

    offs_m = pid_m * BLOCK_M_T + tl.arange(0, BLOCK_M_T)
    offs_n = tl.arange(0, BLOCK_N_T)
    offs_d = tl.arange(0, BLOCK_D_T)

    q_ptrs = (
        Q
        + b * q_stride_b
        + h * q_stride_h
        + offs_m[:, None] * q_stride_m
        + offs_d[None, :] * q_stride_d
    )
    q_mask = (offs_m[:, None] < total_seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full((BLOCK_M_T,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_T, BLOCK_D_T), dtype=tl.float32)

    row_base = layout_h * csr_row_stride_h + pid_m
    csr_start = tl.load(layout_csr_row_indices + row_base)
    csr_end = tl.load(layout_csr_row_indices + row_base + 1)

    qk_scale = softmax_scale * 1.4426950408889634
    csr_iter = csr_start

    while csr_iter < csr_end:
        col_block = tl.load(layout_csr_col_indices + layout_h * csr_col_stride_h + csr_iter)
        start_n = col_block * BLOCK_N_T
        offs_cols = start_n + offs_n

        k_ptrs = (
            K
            + b * k_stride_b
            + h_kv * k_stride_h
            + offs_cols[None, :] * k_stride_m
            + offs_d[:, None] * k_stride_d
        )
        k_mask = (offs_cols[None, :] < total_seq_len) & (offs_d[:, None] < head_dim)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")

        qk = tl.dot(q, k)
        qk = qk * qk_scale

        causal_mask = (offs_m[:, None] >= offs_cols[None, :]) & (
            offs_m[:, None] < total_seq_len
        ) & (offs_cols[None, :] < total_seq_len)
        qk = tl.where(causal_mask, qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = (
            V
            + b * v_stride_b
            + h_kv * v_stride_h
            + offs_cols[:, None] * v_stride_m
            + offs_d[None, :] * v_stride_d
        )
        v_mask = (offs_cols[:, None] < total_seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0, eviction_policy="evict_last")

        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new
        csr_iter += 1

    acc = acc / l_i[:, None]

    o_ptrs = (
        O
        + b * o_stride_b
        + h * o_stride_h
        + offs_m[:, None] * o_stride_m
        + offs_d[None, :] * o_stride_d
    )
    o_mask = (offs_m[:, None] < total_seq_len) & (offs_d[None, :] < head_dim)
    tl.store(o_ptrs, acc, mask=o_mask)


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

    grid = (triton.cdiv(int(total_seq_len), bm), Q.shape[0] * int(num_heads))

    _block_sparse_attention_kernel[grid](
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
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
