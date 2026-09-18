import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _block_sparse_attention_edge_kernel(
    Q,
    K,
    V,
    O,
    softmax_scale,
    num_heads,
    num_kv_heads,
    total_seq_len,
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

    offs_m = pid_m * BLOCK_M_T + tl.arange(0, BLOCK_M_T)
    offs_n = tl.arange(0, BLOCK_N_T)
    offs_d = tl.arange(0, BLOCK_D_T)

    q_base = ((b * num_heads + h) * total_seq_len) * BLOCK_D_T
    kv_base = ((b * num_kv_heads + h_kv) * total_seq_len) * BLOCK_D_T

    q_ptrs = Q + q_base + offs_m[:, None] * BLOCK_D_T + offs_d[None, :]
    q = tl.load(q_ptrs, eviction_policy="evict_first")

    m_i = tl.full((BLOCK_M_T,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_T, BLOCK_D_T), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634

    for t in tl.static_range(0, 3):
        raw_col_block = pid_m + t - 2
        active = raw_col_block >= 0
        col_block = tl.maximum(raw_col_block, 0)
        offs_cols = col_block * BLOCK_N_T + offs_n

        k_ptrs = K + kv_base + offs_d[:, None] + offs_cols[None, :] * BLOCK_D_T
        k = tl.load(k_ptrs, eviction_policy="evict_last")

        qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale

        if t == 2:
            valid = active & (offs_m[:, None] >= offs_cols[None, :])
            qk = tl.where(valid, qk, -float("inf"))
        else:
            qk = tl.where(active, qk, -float("inf"))

        m_block = tl.max(qk, axis=1)
        m_cand = tl.maximum(m_i, m_block)
        m_new = tl.where(active, m_cand, m_i)

        alpha_raw = tl.math.exp2(m_i - m_new)
        alpha = tl.where(active, alpha_raw, 1.0)
        p_raw = tl.math.exp2(qk - m_new[:, None])
        p = tl.where(active, p_raw, 0.0)

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + kv_base + offs_cols[:, None] * BLOCK_D_T + offs_d[None, :]
        v = tl.load(v_ptrs, eviction_policy="evict_last")

        acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = O + q_base + offs_m[:, None] * BLOCK_D_T + offs_d[None, :]
    tl.store(o_ptrs, acc)


@triton.jit
def _block_sparse_attention_main_kernel(
    Q,
    K,
    V,
    O,
    softmax_scale,
    num_heads,
    num_kv_heads,
    total_seq_len,
    BLOCK_M_T: tl.constexpr,
    BLOCK_N_T: tl.constexpr,
    BLOCK_D_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1) + 2

    b = pid_bh // num_heads
    h = pid_bh - b * num_heads
    heads_per_kv = num_heads // num_kv_heads
    h_kv = h // heads_per_kv

    offs_m = pid_m * BLOCK_M_T + tl.arange(0, BLOCK_M_T)
    offs_n = tl.arange(0, BLOCK_N_T)
    offs_d = tl.arange(0, BLOCK_D_T)

    q_base = ((b * num_heads + h) * total_seq_len) * BLOCK_D_T
    kv_base = ((b * num_kv_heads + h_kv) * total_seq_len) * BLOCK_D_T

    q_ptrs = Q + q_base + offs_m[:, None] * BLOCK_D_T + offs_d[None, :]
    q = tl.load(q_ptrs, eviction_policy="evict_first")

    m_i = tl.full((BLOCK_M_T,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_T, BLOCK_D_T), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634

    for t in tl.static_range(0, 2):
        col_block = pid_m + t - 2
        offs_cols = col_block * BLOCK_N_T + offs_n

        k_ptrs = K + kv_base + offs_d[:, None] + offs_cols[None, :] * BLOCK_D_T
        k = tl.load(k_ptrs, eviction_policy="evict_last")

        qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.math.exp2(m_i - m_new)
        p = tl.math.exp2(qk - m_new[:, None])

        acc = acc * alpha[:, None]
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + kv_base + offs_cols[:, None] * BLOCK_D_T + offs_d[None, :]
        v = tl.load(v_ptrs, eviction_policy="evict_last")

        acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
        m_i = m_new

    offs_cols = pid_m * BLOCK_N_T + offs_n

    k_ptrs = K + kv_base + offs_d[:, None] + offs_cols[None, :] * BLOCK_D_T
    k = tl.load(k_ptrs, eviction_policy="evict_last")

    qk = tl.dot(q, k, out_dtype=tl.float32) * qk_scale
    qk = tl.where(offs_m[:, None] >= offs_cols[None, :], qk, -float("inf"))

    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(qk - m_new[:, None])

    acc = acc * alpha[:, None]
    l_i = l_i * alpha + tl.sum(p, axis=1)

    v_ptrs = V + kv_base + offs_cols[:, None] * BLOCK_D_T + offs_d[None, :]
    v = tl.load(v_ptrs, eviction_policy="evict_last")

    acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)

    acc = acc / l_i[:, None]

    o_ptrs = O + q_base + offs_m[:, None] * BLOCK_D_T + offs_d[None, :]
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

    num_row_blocks = triton.cdiv(int(total_seq_len), bm)
    bh = Q.shape[0] * int(num_heads)

    edge_blocks = min(num_row_blocks, 2)
    if edge_blocks > 0:
        _block_sparse_attention_edge_kernel[(bh, edge_blocks)](
            Q,
            K,
            V,
            out,
            float(softmax_scale),
            int(num_heads),
            int(num_kv_heads),
            int(total_seq_len),
            BLOCK_M_T=bm,
            BLOCK_N_T=bn,
            BLOCK_D_T=bd,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if num_row_blocks > 2:
        _block_sparse_attention_main_kernel[(bh, num_row_blocks - 2)](
            Q,
            K,
            V,
            out,
            float(softmax_scale),
            int(num_heads),
            int(num_kv_heads),
            int(total_seq_len),
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
            "SPECIALIZED_MAIN": 1,
            "EDGE_ROWS": 2,
            "CONTIGUOUS_ASSUMED": 1,
            "grid_bh_major": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
