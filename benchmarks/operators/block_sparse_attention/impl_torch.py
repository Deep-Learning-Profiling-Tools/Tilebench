import torch
import math

# Query-row chunk size for the dense reference. Bounds peak memory so the
# reference never materializes the full [B, H, M, N] float32 score/prob
# tensors at once (which OOMs the shared unified memory on GB10 for large M).
# See tilebench_run/BLOCK_SPARSE_ATTENTION_FIX.md for details.
_REF_QUERY_CHUNK = 1024


def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS):
    """
    Reference PyTorch implementation for Block Sparse Attention.

    Memory-safe rewrite: numerically identical to a dense masked attention, but
    computed in query-row chunks so peak memory stays bounded (previously the
    full [B, H, M, N] float32 tensors were allocated at once, ~30 GB at M=10240,
    which crashed the machine on GB10's unified memory).
    """
    B, H, M, D = Q.shape
    _, H_kv, N, _ = K.shape

    # 1. Handle GQA (Grouped Query Attention) by repeating K and V
    head_groups = H // H_kv
    # [B, H_kv, N, D] -> [B, H_kv, head_groups, N, D] -> [B, H, N, D]
    K_expanded = K.unsqueeze(2).expand(B, H_kv, head_groups, N, D).reshape(B, H, N, D)
    V_expanded = V.unsqueeze(2).expand(B, H_kv, head_groups, N, D).reshape(B, H, N, D)

    # 2. Reconstruct the dense block-sparse mask from CSR representation.
    #    Shape [H, M, N] float32 (~3.4 GB at M=N=10240); resident but modest.
    num_rows = math.ceil(M / BLOCK_M)

    sparse_mask = torch.full((H, M, N), float('-inf'), device=Q.device, dtype=torch.float32)
    for h in range(H):
        layout_h = h % num_layout
        for r in range(num_rows):
            start_l = layout_csr_row_indices[layout_h * layout_csr_row_stride_h + r].item()
            end_l = layout_csr_row_indices[layout_h * layout_csr_row_stride_h + r + 1].item()
            for l in range(start_l, end_l):
                c = layout_csr_col_indices[layout_h * layout_csr_col_stride_h + l].item()
                r_start, r_end = r * BLOCK_M, min((r + 1) * BLOCK_M, M)
                c_start, c_end = c * BLOCK_N, min((c + 1) * BLOCK_N, N)
                sparse_mask[h, r_start:r_end, c_start:c_end] = 0.0

    # 3. Dense attention computed in query-row chunks (bounded memory).
    #    Matmuls run in the input dtype (fp16 tensor cores, fp32 accumulate)
    #    instead of the old K/V/Q .float() copies that forced every GEMM onto
    #    the fp32 SIMT path; the softmax runs in fp32 via mask promotion,
    #    matching the DSL kernels' compute layout.
    K_t = K_expanded.transpose(-2, -1)                   # [B, H, D, N]
    cols = torch.arange(N, device=Q.device).view(1, -1)  # [1, N]
    out = torch.empty((B, H, M, D), device=Q.device, dtype=Q.dtype)

    for i0 in range(0, M, _REF_QUERY_CHUNK):
        i1 = min(i0 + _REF_QUERY_CHUNK, M)
        q_chunk = Q[:, :, i0:i1, :]                            # [B, H, c, D]
        scores = torch.matmul(q_chunk, K_t) * softmax_scale    # [B, H, c, N]

        # Causal mask for this chunk of query rows.
        rows = torch.arange(i0, i1, device=Q.device).view(-1, 1)          # [c, 1]
        causal = torch.where(rows >= cols, 0.0, float('-inf')).to(torch.float32)  # [c, N]

        # fp16 scores + fp32 masks promote to fp32 inside the fused add.
        scores = scores + sparse_mask[:, i0:i1, :].unsqueeze(0) + causal
        probs = torch.softmax(scores, dim=-1).to(Q.dtype)
        out[:, :, i0:i1, :] = torch.matmul(probs, V_expanded)

        del scores, probs, q_chunk, causal

    return out
