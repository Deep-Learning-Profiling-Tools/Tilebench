import math

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

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


_xla_cache = torch.utils.weak.WeakTensorKeyDictionary()


def _xla_block_tables(layout_csr_row_indices, layout_csr_col_indices,
                      layout_csr_row_stride_h, layout_csr_col_stride_h,
                      num_layout, M, N, BLOCK_M, BLOCK_N, device):
    num_rows = M // BLOCK_M
    row_ptr_all = layout_csr_row_indices.cpu().long()
    col_all = layout_csr_col_indices.cpu().long()
    rows = []
    for lh in range(num_layout):
        rp = row_ptr_all[lh * layout_csr_row_stride_h:lh * layout_csr_row_stride_h + num_rows + 1]
        cb = lh * layout_csr_col_stride_h
        rows.append([col_all[cb + rp[r]:cb + rp[r + 1]].tolist() for r in range(num_rows)])
    max_c = max(1, max(len(c) for lr in rows for c in lr))
    idx = torch.zeros(num_layout, num_rows, max_c, dtype=torch.long)
    keep = torch.zeros(num_layout, num_rows, BLOCK_M, max_c, BLOCK_N, dtype=torch.bool)
    q_pos = torch.arange(BLOCK_M).view(BLOCK_M, 1)
    k_pos = torch.arange(BLOCK_N).view(1, BLOCK_N)
    for lh in range(num_layout):
        for r in range(num_rows):
            for j, c in enumerate(rows[lh][r]):
                idx[lh, r, j] = c
                keep[lh, r, :, j, :] = (r * BLOCK_M + q_pos) >= (c * BLOCK_N + k_pos)
    return idx.to(device), keep.view(num_layout, num_rows, BLOCK_M, max_c * BLOCK_N).to(device)


def _run_xla(Q, K, V, idx, keep, num_layout, softmax_scale, num_heads, num_kv_heads,
             BLOCK_M, BLOCK_N):
    B, H, M, D = Q.shape
    N = K.shape[2]
    num_rows, max_c = idx.shape[1], idx.shape[2]
    groups = num_heads // num_kv_heads
    Kh = K.repeat_interleave(groups, dim=1).float().reshape(B, H, N // BLOCK_N, BLOCK_N, D)
    Vh = V.repeat_interleave(groups, dim=1).float().reshape(B, H, N // BLOCK_N, BLOCK_N, D)
    q = Q.float().reshape(B, H, num_rows, BLOCK_M, D)
    outs = []
    for lh in range(num_layout):
        heads = list(range(lh, num_heads, num_layout))
        flat = idx[lh].reshape(-1)
        hsel = slice(None) if num_layout == 1 else torch.tensor(heads, device=Q.device)
        kh = Kh if num_layout == 1 else Kh.index_select(1, hsel)
        vh = Vh if num_layout == 1 else Vh.index_select(1, hsel)
        qh = q if num_layout == 1 else q.index_select(1, hsel)
        kg = kh.index_select(2, flat).reshape(B, len(heads), num_rows, max_c * BLOCK_N, D)
        vg = vh.index_select(2, flat).reshape(B, len(heads), num_rows, max_c * BLOCK_N, D)
        sc = torch.matmul(qh, kg.transpose(-1, -2)) * softmax_scale
        sc = sc.masked_fill(~keep[lh], float("-inf"))
        outs.append((heads, torch.matmul(torch.softmax(sc, dim=-1), vg)))
    if num_layout == 1:
        o = outs[0][1]
    else:
        order = [h for heads, _ in outs for h in heads]
        o = torch.cat([t for _, t in outs], dim=1)
        o = o.index_select(1, torch.tensor([order.index(h) for h in range(num_heads)], device=Q.device))
    return o.reshape(B, H, M, D).to(Q.dtype)


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS):
    M = Q.shape[2]
    N = K.shape[2]

    if Q.device.type == "xla":
        tables = _xla_cache.get(layout_csr_col_indices)
        if tables is None:
            tables = _xla_block_tables(
                layout_csr_row_indices, layout_csr_col_indices,
                layout_csr_row_stride_h, layout_csr_col_stride_h,
                num_layout, M, N, BLOCK_M, BLOCK_N, Q.device)
            _xla_cache[layout_csr_col_indices] = tables
        return _run_xla(Q, K, V, *tables, num_layout, softmax_scale,
                        num_heads, num_kv_heads, BLOCK_M, BLOCK_N)

    block_mask = _mask_cache.get(layout_csr_col_indices)
    if block_mask is None:
        block_mask = _build_block_mask(
            layout_csr_row_indices, layout_csr_col_indices,
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, num_heads, M, N, BLOCK_M, BLOCK_N)
        _mask_cache[layout_csr_col_indices] = block_mask

    return _flex(Q, K, V, block_mask=block_mask,
                 scale=softmax_scale, enable_gqa=True)
