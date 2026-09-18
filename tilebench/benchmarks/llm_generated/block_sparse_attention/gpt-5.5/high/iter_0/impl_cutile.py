import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _block_sparse_attention_kernel(
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
    MAX_BLOCKS: ConstInt,
):
    row_block = ct.bid(0)
    bh = ct.bid(1)

    b = bh // num_heads
    h = bh - b * num_heads
    heads_per_kv = num_heads // num_kv_heads
    h_kv = h // heads_per_kv
    layout_h = h % num_layout

    row_base = layout_h * csr_row_stride_h + row_block
    csr_start = ct.load(layout_csr_row_indices, index=(row_base,), shape=())
    csr_end = ct.load(layout_csr_row_indices, index=(row_base + 1,), shape=())

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

    for t in range(0, MAX_BLOCKS):
        csr_pos = csr_start + t
        active = csr_pos < csr_end

        col_block = ct.gather(
            layout_csr_col_indices,
            layout_h * csr_col_stride_h + csr_pos,
            padding_value=0,
        )

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

        k_pos = col_block * BLOCK_N_T + offs_n
        valid = active & (q_pos >= k_pos) & (q_pos < total_seq_len) & (k_pos < total_seq_len)
        qk = ct.where(valid, qk, -np.inf)

        m_new = ct.maximum(m_i, ct.max(qk, axis=1, keepdims=True))
        alpha = ct.exp2(m_i - m_new, flush_to_zero=True)
        p = ct.exp2(qk - m_new, flush_to_zero=True)

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

    max_blocks = 3
    occupancy = 2

    grid = (ct.cdiv(int(total_seq_len), bm), Q.shape[0] * int(num_heads), 1)
    kernel = _block_sparse_attention_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        kernel,
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
            max_blocks,
        ),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_D": bd,
            "MAX_BLOCKS": max_blocks,
            "occupancy": occupancy,
        }
    )
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
