import torch
import math

def generate_vector_add_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.randn(n, dtype=dtype, device=device)
    return (x, y)


def generate_sin_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)
def generate_rope_inputs(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32, device='cuda', **kwargs):

    q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device=device)

    half_dim = head_dim // 2
    cos = torch.randn(seq_len, half_dim, dtype=dtype, device=device)
    sin = torch.randn(seq_len, half_dim, dtype=dtype, device=device)
    
    return (q, cos, sin)
def generate_softmax_inputs(n=None, shape=None, dtype=torch.float32, device='cuda'):
    if shape is None:
        if n is None:
            raise ValueError("Must provide 'n' or 'shape' for softmax inputs")

        cols = int(n**0.5)
        rows = n // cols
        shape = (rows, cols)

    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    x = torch.randn(*shape, dtype=dtype, device=device)
    

    return (x,)
def generate_flash_attn_inputs(batch_size, n_heads, seq_len, head_dim, dtype=torch.float16, device='cuda', **kwargs):

    q = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, dtype=dtype, device=device)

    return (q.contiguous(), k.contiguous(), v.contiguous())
def generate_flash_decode_stage2_inputs(n=None, batch=2, heads=8, seq_len=4096, head_dim=128, block_seq=128, dtype=torch.float32, device='cuda', **kwargs):
    num_blocks = (seq_len + block_seq - 1) // block_seq
    
    b_seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    mid_o = torch.randn((batch, heads, num_blocks, head_dim), dtype=dtype, device=device)
    mid_o_lse = torch.randn((batch, heads, num_blocks), dtype=dtype, device=device)

    block_seq_tensor = torch.tensor(block_seq, dtype=torch.int32, device='cpu') 
    
    return (mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
def generate_mat_mul_inputs(M=1024, K=1024, N=1024, dtype=torch.float16, device='cuda', **kwargs):
    # Depending on the PyTorch version, directly initializing float8_e4m3fn with randn might not be supported.
    # The safest way is to generate float32 and cast.
    a = torch.randn((M, K), dtype=torch.float32, device=device).to(dtype)
    b = torch.randn((K, N), dtype=torch.float32, device=device).to(dtype)
    return (a, b)

def generate_mat_mul_int8_inputs(n=None, M=1024, N=1024, K_b=256, device='cuda', **kwargs):
    K = K_b * 4
    
    a = torch.randint(-128, 127, (M, K), dtype=torch.int8, device=device)
    b = torch.randint(0, 255, (K_b, N), dtype=torch.uint8, device=device).to(torch.int8)
    
    return (a, b)

def generate_streamk_matmul_inputs(n=None, M=1024, N=1024, K=1024, dtype=torch.float16, device='cuda', **kwargs):
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    a = torch.randn((M, K), dtype=torch.float32, device=device).to(dtype)
    b = torch.randn((K, N), dtype=torch.float32, device=device).to(dtype)

    return (a, b)
def generate_block_sparse_attention_inputs(n=None, B=2, H=8, M=1024, D=64, H_kv=2, 
                                           BLOCK_M=64, BLOCK_N=64, BLOCK_D=64, NUM_D_BLOCKS=1,
                                           dtype=torch.float16, device='cuda', **kwargs):
    """
    Generate inputs for block sparse attention.
    Creates a simple "Local Window + Causal" sparse CSR layout.
    """
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
        
    Q = torch.randn((B, H, M, D), dtype=dtype, device=device)
    K = torch.randn((B, H_kv, M, D), dtype=dtype, device=device) # N == M
    V = torch.randn((B, H_kv, M, D), dtype=dtype, device=device)
    
    num_layout = 1 # Shared layout for all heads
    num_rows = math.ceil(M / BLOCK_M)
    num_cols = math.ceil(M / BLOCK_N)
    
    layout_csr_row_stride_h = num_rows + 1
    layout_csr_col_stride_h = num_rows * num_cols # Max possible capacity
    
    # We build a causal local window mask
    window_blocks = 2 # Attend to current block and 2 previous blocks
    
    row_ptrs = []
    col_indices =[]
    
    current_ptr = 0
    for r in range(num_rows):
        row_ptrs.append(current_ptr)
        # Start col is max(0, r - window_blocks)
        # End col is r (inclusive, because of causal)
        start_c = max(0, r - window_blocks)
        end_c = r
        for c in range(start_c, end_c + 1):
            col_indices.append(c)
            current_ptr += 1
            
    row_ptrs.append(current_ptr) # Final ptr
    
    # Pad col_indices to required size
    col_indices = col_indices + [0] * (layout_csr_col_stride_h - len(col_indices))
    
    layout_csr_row_indices = torch.tensor(row_ptrs, dtype=torch.int32, device=device)
    layout_csr_col_indices = torch.tensor(col_indices, dtype=torch.int32, device=device)
    
    softmax_scale = 1.0 / math.sqrt(D)
    EVEN_M = (M % BLOCK_M == 0)
    EVEN_N = (M % BLOCK_N == 0)
    
    return (Q, K, V, layout_csr_row_indices, layout_csr_col_indices, 
            layout_csr_row_stride_h, layout_csr_col_stride_h,
            num_layout, softmax_scale, H, H_kv, M, 
            BLOCK_M, EVEN_M, BLOCK_N, EVEN_N, BLOCK_D, NUM_D_BLOCKS)

def generate_swiglu_inputs(batch_size, ncols, dtype=torch.float32, device='cuda'):
    x = torch.randn(batch_size, ncols, dtype=dtype, device=device)
    y = torch.randn(batch_size, ncols, dtype=dtype, device=device)
    return (x, y)


def generate_cross_entropy_inputs(batch_size, num_classes, dtype=torch.float32, device='cuda'):
    logits = torch.randn((batch_size, num_classes), dtype=dtype, device=device)
    targets = torch.randint(0, num_classes, (batch_size,), device=device, dtype=torch.int64)
    return (logits, targets)


def generate_matrix_transpose_inputs(m, n, dtype=torch.float32, device='cuda'):
    x = torch.randn((m, n), dtype=dtype, device=device)
    return (x,)
def generate_dropout_inputs(n, p, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    x_keep = (torch.rand(n, device=device) > p).to(torch.int32)
    return (x, x_keep, p)


def generate_quantized_gemm_inputs(m, n, k, dtype=torch.float32, device='cuda'):
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)
    scale = 0.02
    a_q = torch.clamp((a / scale).round(), -127, 127).to(torch.int8)
    b_q = torch.clamp((b / scale).round(), -127, 127).to(torch.int8)
    return (a_q, b_q, scale)


def generate_streamk_scheduling_inputs(m, n, k, dtype=torch.float32, device='cuda'):
    a = torch.randn((m, k), dtype=dtype, device=device)
    b = torch.randn((k, n), dtype=dtype, device=device)
    return (a, b)


def generate_packing_values_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_unpacking_values_inputs(n, dtype=torch.float16, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)


def generate_divergence_metric_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.randn(n, dtype=dtype, device=device)
    eps = 1e-5
    return (x, y, eps)


def generate_generic_fused_container_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    gate = torch.randn(n, dtype=dtype, device=device)
    bias = torch.randn(n, dtype=dtype, device=device)
    return (x, gate, bias)


# Registry for input generators
GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "sin": generate_sin_inputs,
    "rope": generate_rope_inputs,
    "flash_attention": generate_flash_attn_inputs,
    "softmax": generate_softmax_inputs,
    "flash_decode": generate_flash_decode_stage2_inputs,
    "matmul_fp16_fp8": generate_mat_mul_inputs,
    "matmul_int8": generate_mat_mul_int8_inputs,
    "streamk_matmul": generate_streamk_matmul_inputs,
    "block_sparse_attention": generate_block_sparse_attention_inputs,
    "cross_entropy": generate_cross_entropy_inputs,
    "matrix_transpose": generate_matrix_transpose_inputs,
    "swiglu": generate_swiglu_inputs,
    "dropout": generate_dropout_inputs,
    "quantized_gemm": generate_quantized_gemm_inputs,
    "streamk_scheduling": generate_streamk_scheduling_inputs,
    "packing_values": generate_packing_values_inputs,
    "unpacking_values": generate_unpacking_values_inputs,
    "divergence_metric": generate_divergence_metric_inputs,
    "generic_fused_container": generate_generic_fused_container_inputs,
}

def get_generator(operator_name):
    if operator_name not in GENERATORS:
        raise ValueError(f"No input generator registered for operator: {operator_name}")
    return GENERATORS[operator_name]
