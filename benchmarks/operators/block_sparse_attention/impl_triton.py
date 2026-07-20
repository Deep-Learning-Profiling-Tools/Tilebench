import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

_DEFAULT_CONFIG = {"num_warps": 4, "num_stages": 2}


@triton.jit
def block_sparse_attention_kernel(
    out_desc,   # output descriptor over [B, H, M, D]
    q_desc,     # query  descriptor over [B, H, M, D]
    k_desc,     # key    descriptor over [B, H_kv, N, D]
    v_desc,     # value  descriptor over [B, H_kv, N, D]
    layout_csr_row_indices,  # block mask CSR format. Shape is [L, num_rows + 1] where num_rows = max_seq_len / BLOCK_M
    layout_csr_col_indices,  # block mask CSR format. Shape is [L, num_rows * num_cols] where num_cols = max_seq_len / BLOCK_N
    layout_csr_row_stride_h,  # stride per head for csr_row_indices, i.e. num_rows + 1
    layout_csr_col_stride_h,  # stride per head for csr_col_indices, i.e. num_rows * num_cols
    num_layout: tl.constexpr,  # number of sparse layout (L)
    softmax_scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    total_seq_len: tl.constexpr,  # Total sequence length including past sequence length and query sequence length.
    BLOCK_M: tl.constexpr,  # block size for q_seq_len
    EVEN_M: tl.constexpr,  # whether q_seq_len % BLOCK_M == 0
    BLOCK_N: tl.constexpr,  # block size for k_seq_len
    EVEN_N: tl.constexpr,  # whether k_seq_len % BLOCK_N == 0
    BLOCK_D: tl.constexpr,  # block size for D
    NUM_D_BLOCKS: tl.constexpr,  # number of data blocks =  D / BLOCK_D
):
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

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    # Initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    if NUM_D_BLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Load q: it will stay in SRAM throughout. TMA load [1,1,BLOCK_M,BLOCK_D] -> [BLOCK_M,BLOCK_D].
    q = q_desc.load([off_b, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, BLOCK_D])
    if NUM_D_BLOCKS >= 2:
        q2 = q_desc.load([off_b, off_h, start_m * BLOCK_M, BLOCK_D]).reshape([BLOCK_M, BLOCK_D])

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
        # K tile [BLOCK_N, BLOCK_D] transposed to [BLOCK_D, BLOCK_N] for tl.dot.
        k = tl.trans(k_desc.load([off_b, off_h_kv, start_n, 0]).reshape([BLOCK_N, BLOCK_D]))
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_D_BLOCKS >= 2:
            k = tl.trans(k_desc.load([off_b, off_h_kv, start_n, BLOCK_D]).reshape([BLOCK_N, BLOCK_D]))
            qk += tl.dot(q2, k)

        qk *= softmax_scale

        # Prompt-only causal + sequence mask.  Keep the explicit validity mask
        # so fully masked rows can be handled without NaNs.
        valid = (
            (offs_m[:, None] < total_seq_len)
            & ((start_n + offs_n[None, :]) < total_seq_len)
            & (offs_m[:, None] >= (start_n + offs_n[None, :]))
        )
        qk = tl.where(valid, qk, float("-inf"))

        # Standard unnormalized online-softmax recurrence, shared verbatim with
        # cuTile.  This needs only one guard for a previously-empty row and one
        # for a fully-masked current block.
        has_prev = l_i > 0.0
        has_valid = tl.sum(valid.to(tl.int32), axis=1) > 0
        block_max = tl.where(has_valid, tl.max(qk, axis=1), float("-inf"))
        m_i_new = tl.maximum(m_i, block_max)
        has_any = has_prev | has_valid
        m_safe = tl.where(has_any, m_i_new, 0.0)
        alpha = tl.where(has_prev, tl.exp(m_i - m_safe), 0.0)
        p = tl.where(valid, tl.exp(qk - m_safe[:, None]), 0.0)
        l_i_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None]
        if NUM_D_BLOCKS >= 2:
            acc2 = acc2 * alpha[:, None]
        p = p.to(q.dtype)

        v = v_desc.load([off_b, off_h_kv, start_n, 0]).reshape([BLOCK_N, BLOCK_D])
        acc += tl.dot(p, v)

        if NUM_D_BLOCKS >= 2:
            v = v_desc.load([off_b, off_h_kv, start_n, BLOCK_D]).reshape([BLOCK_N, BLOCK_D])
            acc2 += tl.dot(p, v)

        l_i = l_i_new
        m_i = tl.where(has_any, m_i_new, m_i)

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    out_desc.store([off_b, off_h, start_m * BLOCK_M, 0], acc.to(out_desc.dtype).reshape([1, 1, BLOCK_M, BLOCK_D]))
    if NUM_D_BLOCKS >= 2:
        acc2 = acc2 / l_safe[:, None]
        out_desc.store([off_b, off_h, start_m * BLOCK_M, BLOCK_D], acc2.to(out_desc.dtype).reshape([1, 1, BLOCK_M, BLOCK_D]))


_block_sparse_attention_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=["total_seq_len"],
    warmup=1,
    rep=3,
)(block_sparse_attention_kernel)


def _make_descriptors(Q, K, V, out, BLOCK_M, BLOCK_N, BLOCK_D):
    """Build host-side TMA descriptors over the 4D [B, H, S, D] tensors.

    Descriptors are created here (in run) using the host-side
    TensorDescriptor API, so descriptor construction is hoisted out of every CTA
    (bowen/fix/operator-matmul_fp32_fp16_fp8 pattern)."""
    q_desc = TensorDescriptor.from_tensor(Q, [1, 1, BLOCK_M, BLOCK_D])
    k_desc = TensorDescriptor.from_tensor(K, [1, 1, BLOCK_N, BLOCK_D])
    v_desc = TensorDescriptor.from_tensor(V, [1, 1, BLOCK_N, BLOCK_D])
    out_desc = TensorDescriptor.from_tensor(out, [1, 1, BLOCK_M, BLOCK_D])
    return q_desc, k_desc, v_desc, out_desc


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
        block_size: int = None, autotune: bool = False):
    """
    Wrapper function to launch the Triton Block Sparse Attention kernel.
    """
    # TMA descriptors require contiguous inputs.
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    q_seq_len = total_seq_len
    batch_size = Q.shape[0]

    grid = (triton.cdiv(q_seq_len, BLOCK_M), batch_size * num_heads)

    out = torch.empty((batch_size, num_heads, q_seq_len, Q.shape[-1]), device=Q.device, dtype=Q.dtype)

    q_desc, k_desc, v_desc, out_desc = _make_descriptors(Q, K, V, out, BLOCK_M, BLOCK_N, BLOCK_D)

    if autotune:
        _block_sparse_attention_kernel_autotuned[grid](
            out_desc, q_desc, k_desc, v_desc,
            layout_csr_row_indices, layout_csr_col_indices,
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, softmax_scale,
            num_heads, num_kv_heads,
            total_seq_len,
            BLOCK_M=BLOCK_M, EVEN_M=EVEN_M,
            BLOCK_N=BLOCK_N, EVEN_N=EVEN_N,
            BLOCK_D=BLOCK_D, NUM_D_BLOCKS=NUM_D_BLOCKS,
        )
    else:
        cfg = _DEFAULT_CONFIG
        block_sparse_attention_kernel[grid](
            out_desc, q_desc, k_desc, v_desc,
            layout_csr_row_indices, layout_csr_col_indices,
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, softmax_scale,
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
