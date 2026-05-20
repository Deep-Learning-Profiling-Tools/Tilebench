import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"num_warps": 4, "num_stages": 2}


@triton.jit
def block_sparse_attention_kernel(
    out,  # output [B, H, M, D]. Note that B is batch_size, H is num_heads, M is q_seq_len, and D is head_size
    Q,  # query [B, H, M, D]
    K,  # key [B, H_kv, N, D]. Note that N is max_seq_len for kv cache, H_kv is num_kv_heads
    V,  # value [B, H_kv, N, D]
    layout_csr_row_indices,  # block mask CSR format. Shape is [L, num_rows + 1] where num_rows = max_seq_len / BLOCK_M
    layout_csr_col_indices,  # block mask CSR format. Shape is [L, num_rows * num_cols] where num_cols = max_seq_len / BLOCK_N
    layout_csr_row_stride_h,  # stride per head for csr_row_indices, i.e. num_rows + 1
    layout_csr_col_stride_h,  # stride per head for csr_col_indices, i.e. num_rows * num_cols
    num_layout,  # number of sparse layout (L)
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_ob,
    stride_oh,
    stride_om,
    num_heads,
    num_kv_heads,
    total_seq_len,  # Total sequence length including past sequence length and query sequence length.
    BLOCK_M: tl.constexpr,  # block size for q_seq_len
    EVEN_M: tl.constexpr,  # whether q_seq_len % BLOCK_M == 0
    BLOCK_N: tl.constexpr,  # block size for k_seq_len
    EVEN_N: tl.constexpr,  # whether k_seq_len % BLOCK_N == 0
    BLOCK_D: tl.constexpr,  # block size for D
    NUM_D_BLOCKS: tl.constexpr,  # number of data blocks =  D / BLOCK_D
):
    #tl.static_print(f"{BLOCK_M=} {BLOCK_N=} {BLOCK_D=} {EVEN_M=} {EVEN_N=} {NUM_D_BLOCKS=}")

    # Past sequence length is 0 since this kernel is for prompt only.
    q_seq_len = total_seq_len

    # Grid is [CDiv(q_seq_len, BLOCK_M), batch_size * num_heads]
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads

    # For group query attention, map the query head index to the corresponding one for key and value.
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups

    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h_kv * stride_kh
    v_base = V + off_b * stride_vb + off_h_kv * stride_vh
    o_base = out + off_b * stride_ob + off_h * stride_oh

    q_desc = tl.make_tensor_descriptor(
        q_base,
        shape=[q_seq_len, BLOCK_D * NUM_D_BLOCKS],
        strides=[stride_qm, 1],
        block_shape=[BLOCK_M, BLOCK_D],
    )
    k_desc = tl.make_tensor_descriptor(
        k_base,
        shape=[total_seq_len, BLOCK_D * NUM_D_BLOCKS],
        strides=[stride_kn, 1],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    v_desc = tl.make_tensor_descriptor(
        v_base,
        shape=[total_seq_len, BLOCK_D * NUM_D_BLOCKS],
        strides=[stride_vn, 1],
        block_shape=[BLOCK_N, BLOCK_D],
    )
    out_desc = tl.make_tensor_descriptor(
        o_base,
        shape=[q_seq_len, BLOCK_D * NUM_D_BLOCKS],
        strides=[stride_om, 1],
        block_shape=[BLOCK_M, BLOCK_D],
    )

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    if NUM_D_BLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load q: it will stay in SRAM throughout
    q = q_desc.load([start_m * BLOCK_M, 0])
    if NUM_D_BLOCKS >= 2:
        q2 = q_desc.load([start_m * BLOCK_M, BLOCK_D])

    layout_h = off_h % num_layout

    # This assumes that past sequence length is 0, otherwise need + (past_seq_len + 1) // BLOCK_M.
    layout_ptr = layout_csr_row_indices + layout_h * layout_csr_row_stride_h + start_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + 1).to(tl.int32)

    # Loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_csr_col_indices + layout_h * layout_csr_col_stride_h + col_idx_idx).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # -- compute qk ----
        k = tl.trans(k_desc.load([start_n, 0]))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_D_BLOCKS >= 2:
            k = tl.trans(k_desc.load([start_n, BLOCK_D]))
            qk += tl.dot(q2, k)

        qk *= softmax_scale

        # This assumes that past sequence length is 0, otherwise need offs_m[:, None] + past_seq_len >= ...
        qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        if NUM_D_BLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)
        # update acc
        v = v_desc.load([start_n, 0])
        acc += tl.dot(p, v)

        if NUM_D_BLOCKS >= 2:
            v = v_desc.load([start_n, BLOCK_D])
            acc2 += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    out_desc.store([start_m * BLOCK_M, 0], acc)
    if NUM_D_BLOCKS >= 2:
        out_desc.store([start_m * BLOCK_M, BLOCK_D], acc2)


_block_sparse_attention_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["total_seq_len"],
)(block_sparse_attention_kernel)


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
        block_size: int = None, autotune: bool = False):
    """
    Wrapper function to launch the Triton Block Sparse Attention kernel.
    """
    ensure_tma_available()
    if not Q.is_contiguous():
        Q = Q.contiguous()
    if not K.is_contiguous():
        K = K.contiguous()
    if not V.is_contiguous():
        V = V.contiguous()

    q_seq_len = total_seq_len
    batch_size = Q.shape[0]

    grid = (triton.cdiv(q_seq_len, BLOCK_M), batch_size * num_heads)

    out = torch.empty((batch_size, num_heads, q_seq_len, Q.shape[-1]), device=Q.device, dtype=Q.dtype)

    if autotune:
        _block_sparse_attention_kernel_autotuned[grid](
            out, Q, K, V,
            layout_csr_row_indices, layout_csr_col_indices,
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_heads, num_kv_heads,
            total_seq_len,
            BLOCK_M=BLOCK_M, EVEN_M=EVEN_M,
            BLOCK_N=BLOCK_N, EVEN_N=EVEN_N,
            BLOCK_D=BLOCK_D, NUM_D_BLOCKS=NUM_D_BLOCKS,
        )
    else:
        cfg = _DEFAULT_CONFIG
        block_sparse_attention_kernel[grid](
            out, Q, K, V,
            layout_csr_row_indices, layout_csr_col_indices,
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, softmax_scale,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_heads, num_kv_heads,
            total_seq_len,
            BLOCK_M=BLOCK_M, EVEN_M=EVEN_M,
            BLOCK_N=BLOCK_N, EVEN_N=EVEN_N,
            BLOCK_D=BLOCK_D, NUM_D_BLOCKS=NUM_D_BLOCKS,
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return out

def get_last_config() -> dict | None:
    cfg = getattr(_block_sparse_attention_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}
