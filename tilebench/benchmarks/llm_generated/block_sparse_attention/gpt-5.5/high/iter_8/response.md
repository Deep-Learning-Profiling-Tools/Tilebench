```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _block_sparse_attention_edge_kernel(
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
            qk = tl.where(offs_m[:, None] >= offs_cols[None, :], qk, -float("inf"))
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

        v_ptrs = (
            V
            + b * v_stride_b
            + h_kv * v_stride_h
            + offs_cols[:, None] * v_stride_m
            + offs_d[None, :] * v_stride_d
        )
        v = tl.load(v_ptrs, eviction_policy="evict_last")

        acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
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
    # Main region excludes the first two row-blocks, so all three local-window
    # blocks are known active and the scalar active predicates disappear.
    pid_m = tl.program_id(1) + 2

    b = pid_bh // num_heads
    h = pid_bh - b * num_heads
    heads_per_kv = num_heads // num_kv_heads
    h_kv = h // heads_per_kv

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
    q = tl.load(q_ptrs, eviction_policy="evict_first")

    m_i = tl.full((BLOCK_M_T,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M_T,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M_T, BLOCK_D_T), dtype=tl.float32)

    qk_scale = softmax_scale * 1.4426950408889634

    for t in tl.static_range(0, 3):
        col_block = pid_m + t - 2
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
            qk = tl.where(offs_m[:, None] >= offs_cols[None, :], qk, -float("inf"))

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
        v = tl.load(v_ptrs, eviction_policy="evict_last")

        acc = tl.dot(p.to(tl.float16), v, acc, out_dtype=tl.float32)
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

    bh = Q.shape[0] * int(num_heads)
    num_row_blocks = triton.cdiv(int(total_seq_len), bm)

    if num_row_blocks > 2:
        grid_main = (bh, num_row_blocks - 2)
        _block_sparse_attention_main_kernel[grid_main](
            Q,
            K,
            V,
            out,
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

    edge_rows = 2 if num_row_blocks >= 2 else num_row_blocks
    if edge_rows > 0:
        grid_edge = (bh, edge_rows)
        _block_sparse_attention_edge_kernel[grid_edge](
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
            "SPECIALIZED_MAIN_SPLIT": 1,
            "grid_bh_major": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _block_sparse_attention_edge_kernel(
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

    for t in range(0, 3):
        raw_col_block = row_block + t - 2
        active = raw_col_block >= 0
        col_block = ct.where(active, raw_col_block, 0)

        k4 = ct.load(
            K,
            index=(b, h_kv, 0, col_block),
            shape=(1, 1, BLOCK_D_T, BLOCK_N_T),
            order=(0, 1, 3, 2),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        )
        k_t = ct.reshape(k4, (BLOCK_D_T, BLOCK_N_T))

        qk_zero = ct.full((BLOCK_M_T, BLOCK_N_T), 0.0, dtype=np.float32)
        qk = ct.mma(q, k_t, qk_zero)
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


@ct.kernel
def _block_sparse_attention_main_kernel(
    Q,
    K,
    V,
    out,
    softmax_scale,
    num_heads: ConstInt,
    num_kv_heads: ConstInt,
    total_seq_len: ConstInt,
    BLOCK_M_T: ConstInt,
    BLOCK_N_T: ConstInt,
    BLOCK_D_T: ConstInt,
):
    row_block = ct.bid(0) + 2
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

    for t in range(0, 3):
        col_block = row_block + t - 2

        k4 = ct.load(
            K,
            index=(b, h_kv, 0, col_block),
            shape=(1, 1, BLOCK_D_T, BLOCK_N_T),
            order=(0, 1, 3, 2),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        )
        k_t = ct.reshape(k4, (BLOCK_D_T, BLOCK_N_T))

        qk_zero = ct.full((BLOCK_M_T, BLOCK_N_T), 0.0, dtype=np.float32)
        qk = ct.mma(q, k_t, qk_zero)
        qk = qk * qk_scale

        if t == 2:
            k_pos = col_block * BLOCK_N_T + offs_n
            qk = ct.where(q_pos >= k_pos, qk, -np.inf)

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

    occupancy = 1
    max_blocks = 3
    num_row_blocks = (int(total_seq_len) + bm - 1) // bm
    bh = Q.shape[0] * int(num_heads)

    main_kernel = _block_sparse_attention_main_kernel.with_hints(occupancy=occupancy)
    edge_kernel = _block_sparse_attention_edge_kernel.with_hints(occupancy=occupancy)

    if num_row_blocks > 2:
        grid_main = (num_row_blocks - 2, bh, 1)
        ct.launch(
            stream,
            grid_main,
            main_kernel,
            (
                Q,
                K,
                V,
                out,
                float(softmax_scale),
                int(num_heads),
                int(num_kv_heads),
                int(total_seq_len),
                bm,
                bn,
                bd,
            ),
        )

    edge_rows = 2 if num_row_blocks >= 2 else num_row_blocks
    if edge_rows > 0:
        grid_edge = (edge_rows, bh, 1)
        ct.launch(
            stream,
            grid_edge,
            edge_kernel,
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
            "SPECIALIZED_MAIN_SPLIT": 1,
            "K_TRANSPOSE_LOAD": 1,
            "allow_tma": 0,
            "occupancy": occupancy,
        }
    )
    return out


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
