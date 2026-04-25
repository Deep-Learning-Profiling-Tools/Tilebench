import math
from types import SimpleNamespace

import torch
import cuda.tile as ct

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:
    ct_experimental = None

ConstInt = ct.Constant[int]

_last_autotune_config: dict | None = None

_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [1, 2, 4, 8]]


@ct.kernel
def block_sparse_attention_cutile_kernel(
    Out, Q, K, V,
    csr_row_indices, csr_col_indices,
    csr_row_stride_h: ConstInt, csr_col_stride_h: ConstInt,
    num_layout: ConstInt, softmax_scale: ct.Constant[float],
    num_heads: ConstInt, num_kv_heads: ConstInt, total_seq_len: ConstInt,
    BLOCK_M: ConstInt, BLOCK_N: ConstInt, TOTAL_D: ConstInt
):
    start_m = ct.bid(0)
    off_bh = ct.bid(1)

    off_h = off_bh % num_heads
    off_b = off_bh // num_heads

    # GQA mapping: query head -> kv head
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups

    # Load Q tile.
    # Note: ct.load uses tile-space indices, not element-space indices.
    q_tile = ct.load(
        Q,
        index=(off_b, off_h, start_m, 0),
        shape=(1, 1, BLOCK_M, TOTAL_D),
        padding_mode=ct.PaddingMode.ZERO,
    )
    q = ct.reshape(q_tile, (BLOCK_M, TOTAL_D))

    # Online softmax state
    m_i = ct.full((BLOCK_M, 1), -float("inf"), dtype=ct.float32)
    l_i = ct.full((BLOCK_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((BLOCK_M, TOTAL_D), 0.0, dtype=ct.float32)

    # CSR row pointers for this head/layout
    layout_h = off_h % num_layout
    row_idx_ptr = layout_h * csr_row_stride_h + start_m
    start_l = ct.load(csr_row_indices, index=(row_idx_ptr,), shape=())
    end_l = ct.load(csr_row_indices, index=(row_idx_ptr + 1,), shape=())

    # Query row offsets in element space, used only for masking
    offs_m = start_m * BLOCK_M + ct.expand_dims(ct.arange(BLOCK_M, dtype=ct.int32), 1)  # [BLOCK_M, 1]
    valid_m = offs_m < total_seq_len  # [BLOCK_M, 1]

    l = start_l
    while l < end_l:
        col_idx_ptr = layout_h * csr_col_stride_h + l
        col_idx = ct.load(csr_col_indices, index=(col_idx_ptr,), shape=())

        # IMPORTANT:
        # col_idx is already a block-column index in CSR.
        # For ct.load, we must use tile-space index directly.
        start_n_tile = col_idx

        # Load K tile
        k_tile = ct.load(
            K,
            index=(off_b, off_h_kv, start_n_tile, 0),
            shape=(1, 1, BLOCK_N, TOTAL_D),
            padding_mode=ct.PaddingMode.ZERO,
        )
        k = ct.reshape(k_tile, (BLOCK_N, TOTAL_D))
        k_t = ct.transpose(k, 0, 1)

        # QK
        qk = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q, k_t, qk)
        qk = qk * softmax_scale

        # Key offsets in element space, used for causal/seq masking
        offs_n = start_n_tile * BLOCK_N + ct.expand_dims(ct.arange(BLOCK_N, dtype=ct.int32), 0)  # [1, BLOCK_N]
        valid_n = offs_n < total_seq_len  # [1, BLOCK_N]

        causal_mask = offs_m >= offs_n                     # [BLOCK_M, BLOCK_N]
        seq_mask = ct.bitwise_and(valid_m, valid_n)        # [BLOCK_M, BLOCK_N]
        mask = ct.bitwise_and(causal_mask, seq_mask)       # [BLOCK_M, BLOCK_N]

        qk = ct.where(mask, qk, -float("inf"))

        # ---------------------------------------------------------------------
        # Safe online softmax update
        #
        # We must handle rows that are fully masked in the current KV block.
        # Otherwise, max=-inf and exp(-inf - -inf) can produce NaN.
        # ---------------------------------------------------------------------

        # Per-row max for this block
        m_ij = ct.max(qk, axis=1, keepdims=True)  # [BLOCK_M, 1]

        # Whether a row has any valid element in this block
        valid_count = ct.sum(ct.astype(mask, ct.int32), axis=1, keepdims=True)  # [BLOCK_M, 1]
        row_has_valid = valid_count > 0                                           # [BLOCK_M, 1]

        # Whether this row already has previous valid softmax state
        row_has_prev = l_i > 0.0                                                  # [BLOCK_M, 1]

        # Safe versions used to avoid invalid arithmetic like -inf - (-inf)
        m_i_safe = ct.where(row_has_prev, m_i, 0.0)       # [BLOCK_M, 1]
        m_ij_safe = ct.where(row_has_valid, m_ij, 0.0)    # [BLOCK_M, 1]

        # Compute p safely.
        # For invalid positions, qk is -inf, so exp(-inf - finite) = 0.
        # For fully-masked rows, m_ij_safe is forced to 0, so exp(-inf - 0) = 0.
        p = ct.exp(qk - m_ij_safe)                        # [BLOCK_M, BLOCK_N]
        p = ct.where(mask, p, 0.0)

        l_ij = ct.sum(p, axis=1, keepdims=True)           # [BLOCK_M, 1]

        # Candidate new max:
        # - If row has previous state: max(old_max, block_max)
        # - Else: block_max
        m_i_new_candidate = ct.where(
            row_has_prev,
            ct.maximum(m_i_safe, m_ij_safe),
            m_ij_safe,
        )

        # Safe alpha/beta candidates
        # alpha = exp(old_max - new_max) if previous state exists, else 0
        alpha_candidate = ct.where(
            row_has_prev,
            ct.exp(m_i_safe - m_i_new_candidate),
            0.0,
        )

        # beta = exp(block_max - new_max) if current block has valid entries, else 0
        beta_candidate = ct.where(
            row_has_valid,
            ct.exp(m_ij_safe - m_i_new_candidate),
            0.0,
        )

        # Candidate new l_i
        l_i_new_candidate = alpha_candidate * l_i + beta_candidate * l_ij

        # Only update state for rows that have valid entries in this block
        m_i_new = ct.where(row_has_valid, m_i_new_candidate, m_i)
        l_i_new = ct.where(row_has_valid, l_i_new_candidate, l_i)

        # Safe denominator for later divisions
        l_i_new_safe = ct.where(l_i_new > 0.0, l_i_new, 1.0)

        # Normalize probabilities for this block contribution
        p_scale = beta_candidate / l_i_new_safe
        p = p * p_scale

        # Rescale accumulator
        acc_scale_raw = (l_i * alpha_candidate) / l_i_new_safe
        acc_scale = ct.where(row_has_valid, acc_scale_raw, 1.0)
        acc = acc * acc_scale

        # Load V tile
        v_tile = ct.load(
            V,
            index=(off_b, off_h_kv, start_n_tile, 0),
            shape=(1, 1, BLOCK_N, TOTAL_D),
            padding_mode=ct.PaddingMode.ZERO,
        )
        v = ct.reshape(v_tile, (BLOCK_N, TOTAL_D))

        # Accumulate P @ V
        p_casted = ct.astype(p, Q.dtype)
        acc = ct.mma(p_casted, v, acc)

        # Commit online softmax state
        l_i = l_i_new
        m_i = m_i_new

        l = l + 1

    # Store output.
    # ct.store also uses tile-space index, and out-of-bound stores are ignored.
    acc_casted = ct.astype(acc, Out.dtype)
    acc_tile = ct.reshape(acc_casted, (1, 1, BLOCK_M, TOTAL_D))
    ct.store(Out, index=(off_b, off_h, start_m, 0), tile=acc_tile)


def run(
    Q, K, V,
    layout_csr_row_indices, layout_csr_col_indices,
    layout_csr_row_stride_h, layout_csr_col_stride_h,
    num_layout, softmax_scale, num_heads, num_kv_heads,
    total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
    block_size: int = None, autotune: bool = False
):
    global _last_autotune_config
    batch_size = Q.shape[0]
    D = Q.shape[-1]
    TOTAL_D = BLOCK_D * NUM_D_BLOCKS

    assert D == TOTAL_D, (
        f"Invalid config: Q.shape[-1]={D}, BLOCK_D={BLOCK_D}, "
        f"NUM_D_BLOCKS={NUM_D_BLOCKS}, expected D={TOTAL_D}"
    )
    assert K.shape[-1] == D and V.shape[-1] == D, "K/V head dim must match Q head dim"
    assert total_seq_len == Q.shape[2], "This prompt-only kernel expects total_seq_len == Q.shape[2]"
    assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

    out = torch.empty_like(Q)

    # Grid: [num_q_blocks, batch_size * num_heads, 1]
    grid = (math.ceil(total_seq_len / BLOCK_M), batch_size * num_heads, 1)

    stream = torch.cuda.current_stream()
    args = (
        out, Q, K, V,
        layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, float(softmax_scale),
        num_heads, num_kv_heads, total_seq_len,
        BLOCK_M, BLOCK_N, TOTAL_D,
    )

    if autotune and ct_experimental is not None:
        result = ct_experimental.autotune_launch(
            stream,
            grid_fn=lambda cfg: grid,
            kernel=block_sparse_attention_cutile_kernel,
            args_fn=lambda cfg: args,
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
            search_space=_SEARCH_SPACE,
        )
        _last_autotune_config = {"occupancy": result.tuned_config.occupancy}
    else:
        ct.launch(
            stream,
            grid,
            block_sparse_attention_cutile_kernel,
            args
        )

    return out


def get_last_config() -> dict | None:
    return _last_autotune_config
