import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bsa_kernel(
    Q, K, V, Out,
    csr_row_ind, csr_col_ind,
    csr_row_stride_h, csr_col_stride_h, num_layout,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    H, M, head_groups,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    INNER_STAGES: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    off_b = off_bh // H
    off_h = off_bh % H
    off_hkv = off_h // head_groups
    layout_h = off_h % num_layout

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_hkv * stride_kh
    v_offset = off_b * stride_vb + off_hkv * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + q_offset + offs_m[:, None] * stride_qm + offs_d[None, :]
    if EVEN_M:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)

    qk_scale = softmax_scale * 1.44269504  # 1 / ln(2)
    q = (q * qk_scale).to(q.dtype)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    row_idx_ptr = csr_row_ind + layout_h * csr_row_stride_h + start_m
    start_l = tl.load(row_idx_ptr)
    end_l = tl.load(row_idx_ptr + 1)

    col_base = csr_col_ind + layout_h * csr_col_stride_h

    # Software-pipeline the inner CSR-driven loop to overlap K/V loads with
    # the QK matmul + softmax update of the previous iteration.
    for l_idx in tl.range(start_l, end_l, num_stages=INNER_STAGES):
        c = tl.load(col_base + l_idx)
        start_n = c * BLOCK_N
        cols = start_n + offs_n

        k_ptrs = K + k_offset + cols[None, :] * stride_kn + offs_d[:, None]
        if EVEN_N:
            k = tl.load(k_ptrs, eviction_policy="evict_last")
        else:
            k = tl.load(k_ptrs, mask=cols[None, :] < M, other=0.0,
                        eviction_policy="evict_last")

        qk = tl.dot(q, k)
        causal = offs_m[:, None] >= cols[None, :]
        if not EVEN_N:
            causal = causal & (cols[None, :] < M)
        qk = tl.where(causal, qk, float('-inf'))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        v_ptrs = V + v_offset + cols[:, None] * stride_vn + offs_d[None, :]
        if EVEN_N:
            v = tl.load(v_ptrs, eviction_policy="evict_last")
        else:
            v = tl.load(v_ptrs, mask=cols[:, None] < M, other=0.0,
                        eviction_policy="evict_last")

        acc = acc * alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = Out + o_offset + offs_m[:, None] * stride_om + offs_d[None, :]
    if EVEN_M:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty))
    else:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < M)


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS):
    B, H, M, D = Q.shape
    head_groups = H // num_kv_heads

    out = torch.empty_like(Q)
    HEAD_DIM = D

    num_warps = 4
    num_stages = 3
    inner_stages = 2

    grid = (triton.cdiv(M, BLOCK_M), B * H, 1)
    _bsa_kernel[grid](
        Q, K, V, out,
        layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h, num_layout,
        softmax_scale,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H, M, head_groups,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        HEAD_DIM=HEAD_DIM,
        EVEN_M=EVEN_M, EVEN_N=EVEN_N,
        INNER_STAGES=inner_stages,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HEAD_DIM": HEAD_DIM,
        "num_warps": num_warps, "num_stages": num_stages,
        "inner_stages": inner_stages,
        "evict_policy": "evict_last", "prescale_q": True,
    })
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
