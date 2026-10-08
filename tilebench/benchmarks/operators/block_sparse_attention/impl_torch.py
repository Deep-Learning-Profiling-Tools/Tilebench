import math

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# One compiled graph per sweep case (each M is a distinct static shape);
# the default dynamo cache limit (8) silently falls back to the slow eager
# flex path after 8 cases.
torch._dynamo.config.cache_size_limit = 64

_flex = torch.compile(flex_attention, dynamic=False)

_mask_cache = torch.utils.weak.WeakTensorKeyDictionary()


def _build_block_mask(layout_csr_row_indices, layout_csr_col_indices,
                      layout_csr_row_stride_h, layout_csr_col_stride_h,
                      num_layout, num_heads, M, N, BLOCK_M, BLOCK_N):
    device = layout_csr_row_indices.device
    num_rows = math.ceil(M / BLOCK_M)
    num_cols = math.ceil(N / BLOCK_N)

    sel = torch.zeros(num_layout, num_rows, num_cols, dtype=torch.bool, device=device)
    for lh in range(num_layout):
        base = lh * layout_csr_row_stride_h
        row_ptr = layout_csr_row_indices[base:base + num_rows + 1].long()
        counts = row_ptr[1:] - row_ptr[:-1]
        rows = torch.repeat_interleave(
            torch.arange(num_rows, device=device), counts)
        col_base = lh * layout_csr_col_stride_h
        cols = layout_csr_col_indices[
            col_base + row_ptr[0]:col_base + row_ptr[-1]].long()
        sel[lh, rows, cols] = True

    def mask_mod(b, h, q_idx, kv_idx):
        return (
            sel[h % num_layout, q_idx // BLOCK_M, kv_idx // BLOCK_N]
            & (q_idx >= kv_idx)
        )

    return create_block_mask(mask_mod, B=None, H=num_heads,
                             Q_LEN=M, KV_LEN=N, device=device)


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS):
    M = Q.shape[2]
    N = K.shape[2]

    block_mask = _mask_cache.get(layout_csr_col_indices)
    if block_mask is None:
        block_mask = _build_block_mask(
            layout_csr_row_indices, layout_csr_col_indices,
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, num_heads, M, N, BLOCK_M, BLOCK_N)
        _mask_cache[layout_csr_col_indices] = block_mask

    return _flex(Q, K, V, block_mask=block_mask,
                 scale=softmax_scale, enable_gqa=True)
