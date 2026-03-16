import torch
import math

def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices, 
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads, 
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS):
    """
    Reference PyTorch implementation for Block Sparse Attention.
    """
    B, H, M, D = Q.shape
    _, H_kv, N, _ = K.shape
    
    # 1. Handle GQA (Grouped Query Attention) by repeating K and V
    head_groups = H // H_kv
    #[B, H_kv, N, D] -> [B, H_kv, head_groups, N, D] ->[B, H, N, D]
    K_expanded = K.unsqueeze(2).expand(B, H_kv, head_groups, N, D).reshape(B, H, N, D)
    V_expanded = V.unsqueeze(2).expand(B, H_kv, head_groups, N, D).reshape(B, H, N, D)
    
    # 2. Reconstruct the dense attention mask from CSR representation
    num_rows = math.ceil(M / BLOCK_M)
    num_cols = math.ceil(N / BLOCK_N)
    
    # Initialize with -inf (masked out)
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
                
                # Unmask this block
                sparse_mask[h, r_start:r_end, c_start:c_end] = 0.0

    # Expand mask for batch size
    sparse_mask = sparse_mask.unsqueeze(0).expand(B, H, M, N)
    
    # 3. Create Causal Mask (Lower Triangular)
    # The kernel says: qk += tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), 0, float("-inf"))
    causal_mask = torch.tril(torch.ones(M, N, device=Q.device, dtype=torch.bool))
    causal_mask = torch.where(causal_mask, 0.0, float('-inf'))
    
    # 4. Dense Attention Computation
    scores = torch.matmul(Q.float(), K_expanded.float().transpose(-2, -1)) * softmax_scale
    
    # Apply both Sparse Mask and Causal Mask
    scores = scores + sparse_mask + causal_mask.unsqueeze(0).unsqueeze(0)
    
    probs = torch.softmax(scores, dim=-1)
    
    out = torch.matmul(probs, V_expanded.float())
    
    return out.to(Q.dtype)