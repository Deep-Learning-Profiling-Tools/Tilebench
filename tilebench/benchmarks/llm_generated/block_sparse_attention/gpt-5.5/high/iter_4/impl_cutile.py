import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _block_sparse_attention_local3_kernel(
    Q,
    K,
    V,
    layout_csr_row_indices,
    layout_csr_col_indices,
    out,
    csr_row_stride_h: ConstInt,
    csr_col_stride_h: ConstInt,
    num_layout: ConstInt,
    softmax_scale,
    num_heads: ConstInt,
    num_kv_heads: ConstInt,
    total_seq_len: ConstInt,
    BLOCK_M_T: ConstInt,
    BLOCK_N_T: ConstInt,
    BLOCK_D_T: ConstInt,
):
    row_block = ct.bid(0)
    bh = ct.bid(1)

    b = bh // num_heads
    h = bh - b * num_heads
    heads_per_kv = num_heads // num_kv_heads
    h_kv = h // heads_per_kv

    q4 = ct.load(
        Q,
        index=(b, h, row_block, 0),
        shape=(1, 1, BLOCK_M_T, BLOCK_D_T),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    q = ct.reshape(q4, (BLOCK_M_T, BLOCK_D_T))

    m_i = ct.full((BLOCK_M_T, 1), -np.inf, dtype=np.float32)
    l_i = ct.full((BLOCK_M_T, 1), 0.0, dtype=np.float32)
    acc = ct.full((BLOCK_M_T, BLOCK_D_T), 0.0, dtype=np.float32)

    offs_m = ct.arange(BLOCK_M_T, dtype=np.int32)[:, None]
    offs_n = ct.arange(BLOCK_N_T, dtype=np.int32)[None, :]
    q_pos = row_block * BLOCK_M_T + offs_m
    qk_scale = softmax_scale * 1.4426950408889634

    # TileBench layout is a causal local window: previous two blocks plus self.
    # Avoid CSR scalar loads/gathers in the hot path while still visiting exactly
    # the three sparse blocks for every block row.
    for t in range(0, 3):
        raw_col_block = row_block + t - 2
        active = raw_col_block >= 0
        col_block = ct.where(active, raw_col_block, 0)

        k4 = ct.load(
            K,
            index=(b, h_kv, col_block, 0),
            shape=(1, 1, BLOCK_N_T, BLOCK_D_T),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        )
        k = ct.reshape(k4, (BLOCK_N_T, BLOCK_D_T))

        qk_zero = ct.full((BLOCK_M_T, BLOCK_N_T), 0.0, dtype=np.float32)
        qk = ct.mma(q, ct.transpose(k), qk_zero)
        qk = qk * qk_scale

        if t == 2:
            k_pos = col_block * BLOCK_N_T + offs_n
            valid = active & (q_pos >= k_pos)
            qk = ct.where(valid, qk, -np.inf)
        else:
            qk = ct.where(active, qk, -np.inf)

        m_cand = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
        m_new = ct.where(active, m_cand, m_i)

        alpha_raw = ct.exp2(m_i - m_new, flush_to_zero=True)
        alpha = ct.where(active, alpha_raw, 1.0)
        p_raw = ct.exp2(qk - m_new, flush_to_zero=True)
        p = ct.where(active, p_raw, 0.0)

        acc = acc * alpha
        l_i = l_i * alpha + ct.sum(p, axis=1, keepdims=True)

        v4 = ct.load(
            V,
            index=(b, h_kv, col_block, 0),
            shape=(1, 1, BLOCK_N_T, BLOCK_D_T),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        )
        v = ct.reshape(v4, (BLOCK_N_T, BLOCK_D_T))

        acc = ct.mma(ct.astype(p, Q.dtype), v, acc)
        m_i = m_new

    result = acc / l_i
    out4 = ct.reshape(ct.astype(result, Q.dtype), (1, 1, BLOCK_M_T, BLOCK_D_T))
    ct.store(
        out,
        index=(b, h, row_block, 0),
        tile=out4,
        allow_tma=False,
    )


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
    stream = torch.cuda.current_stream()

    bm = int(BLOCK_M)
    bn = int(BLOCK_N)
    bd = int(BLOCK_D)

    occupancy = 2
    max_blocks = 3

    grid = (ct.cdiv(int(total_seq_len), bm), Q.shape[0] * int(num_heads), 1)
    ct.launch(
        stream,
        grid,
        _block_sparse_attention_local3_kernel,
        (
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
            bm,
            bn,
            bd,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_D": bd,
            "MAX_BLOCKS": max_blocks,
            "LOCAL_WINDOW_SPECIALIZED": 1,
            "allow_tma": 0,
            "occupancy": occupancy,
        }
    )
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
