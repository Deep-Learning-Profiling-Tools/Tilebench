import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]

@ct.kernel
def block_sparse_attention_cutile_kernel(
    Out, Q, K, V,
    csr_row_indices, csr_col_indices,
    csr_row_stride_h: ConstInt, csr_col_stride_h: ConstInt,
    num_layout: ConstInt, softmax_scale: ct.Constant[float],
    num_heads: ConstInt, num_kv_heads: ConstInt, total_seq_len: ConstInt,
    BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_D: ConstInt
):
    start_m = ct.bid(0)
    off_bh = ct.bid(1)
    
    off_h = off_bh % num_heads
    off_b = off_bh // num_heads
    
    # GQA mapping
    head_groups = num_heads // num_kv_heads
    off_h_kv = off_h // head_groups
    
    # Load Q tile and reshape to 2D
    q_tile = ct.load(Q, index=(off_b, off_h, start_m, 0), shape=(1, 1, BLOCK_M, BLOCK_D), padding_mode=ct.PaddingMode.ZERO)
    q_tile_2d = ct.reshape(q_tile, (BLOCK_M, BLOCK_D))
    
    # Initialize Online Softmax states
    m_i = ct.full((BLOCK_M, 1), -float('inf'), dtype=ct.float32)
    l_i = ct.full((BLOCK_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((BLOCK_M, BLOCK_D), 0.0, dtype=ct.float32)
    
    # Fetch CSR pointers
    layout_h = off_h % num_layout
    row_idx_ptr = layout_h * csr_row_stride_h + start_m
    start_l = ct.load(csr_row_indices, index=(row_idx_ptr,), shape=())
    end_l = ct.load(csr_row_indices, index=(row_idx_ptr + 1,), shape=())
    
    # Setup row offsets for masking
    offs_m = start_m * BLOCK_M + ct.expand_dims(ct.arange(BLOCK_M, dtype=ct.int32), 1)
    valid_m = offs_m < total_seq_len
    
    # Sparse Loop
    l = start_l
    while l < end_l:
        col_idx_ptr = layout_h * csr_col_stride_h + l
        col_idx = ct.load(csr_col_indices, index=(col_idx_ptr,), shape=())
        start_n = col_idx
        
        # Load K and compute QK
        k_tile = ct.load(K, index=(off_b, off_h_kv, start_n, 0), shape=(1, 1, BLOCK_N, BLOCK_D), padding_mode=ct.PaddingMode.ZERO)
        k_tile_2d = ct.reshape(k_tile, (BLOCK_N, BLOCK_D))
        k_tile_T = ct.transpose(k_tile_2d, 0, 1)
        
        qk = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
        qk = ct.mma(q_tile_2d, k_tile_T, qk)
        qk = qk * softmax_scale
        
        # Causal & Sequence Masking
        offs_n = start_n * BLOCK_N + ct.expand_dims(ct.arange(BLOCK_N, dtype=ct.int32), 0)
        valid_n = offs_n < total_seq_len
        causal_mask = offs_m >= offs_n
        seq_mask = ct.bitwise_and(valid_m, valid_n)
        mask = ct.bitwise_and(causal_mask, seq_mask)
        
        qk = ct.where(mask, qk, -float('inf'))
        
        # Online Softmax Math
        m_ij = ct.max(qk, axis=1)
        m_ij = ct.expand_dims(m_ij, 1)
        
        m_i_new = ct.maximum(m_i, m_ij)
        alpha = ct.exp(m_i - m_i_new)
        beta = ct.exp(m_ij - m_i_new)
        
        p = ct.exp(qk - m_ij)
        l_ij = ct.sum(p, axis=1)
        l_ij = ct.expand_dims(l_ij, 1)
        
        l_i_new = alpha * l_i + beta * l_ij
        
        p_scale = beta / l_i_new
        p = p * p_scale
        
        acc_scale = (l_i / l_i_new) * alpha
        acc = acc * acc_scale
        
        # Downcast P and multiply with V
        p_casted = ct.astype(p, Q.dtype)
        
        v_tile = ct.load(V, index=(off_b, off_h_kv, start_n, 0), shape=(1, 1, BLOCK_N, BLOCK_D), padding_mode=ct.PaddingMode.ZERO)
        v_tile_2d = ct.reshape(v_tile, (BLOCK_N, BLOCK_D))
        
        acc = ct.mma(p_casted, v_tile_2d, acc)
        
        # Update states
        l_i = l_i_new
        m_i = m_i_new
        
        l = l + 1
        
    # Write back result
    acc_casted = ct.astype(acc, Out.dtype)
    acc_reshaped = ct.reshape(acc_casted, (1, 1, BLOCK_M, BLOCK_D))
    ct.store(Out, index=(off_b, off_h, start_m, 0), tile=acc_reshaped)

def run(Q, K, V, layout_csr_row_indices, layout_csr_col_indices,
        layout_csr_row_stride_h, layout_csr_col_stride_h,
        num_layout, softmax_scale, num_heads, num_kv_heads,
        total_seq_len, BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS,
        block_size: int = None, autotune: bool = False):
    
    batch_size = Q.shape[0]
    out = torch.empty_like(Q)
    
    # Grid:[M_blocks, Batch * Heads, 1]
    grid = (math.ceil(total_seq_len / BLOCK_M), batch_size * num_heads, 1)
    
    # cuTile handles D dimension directly. 
    # If Triton used NUM_D_BLOCKS=2 (e.g. D=128, BLOCK_D=64), we can just tell cuTile to load 128.
    ACTUAL_BLOCK_D = BLOCK_D * NUM_D_BLOCKS
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        block_sparse_attention_cutile_kernel,
        (out, Q, K, V, 
         layout_csr_row_indices, layout_csr_col_indices,
         layout_csr_row_stride_h, layout_csr_col_stride_h,
         num_layout, float(softmax_scale),
         num_heads, num_kv_heads, total_seq_len,
         BLOCK_M, BLOCK_N, ACTUAL_BLOCK_D)
    )
    
    return out

def get_last_config() -> dict | None:
    return None