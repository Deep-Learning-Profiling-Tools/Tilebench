import torch

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
    # 计算 Num Blocks
    num_blocks = (seq_len + block_seq - 1) // block_seq
    
    b_seqlen = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    mid_o = torch.randn((batch, heads, num_blocks, head_dim), dtype=dtype, device=device)
    mid_o_lse = torch.randn((batch, heads, num_blocks), dtype=dtype, device=device)
    
    # 【修改点】将 block_seq 包装成 Tensor 放入返回列表
    # 这样 engine 就会把它传给 run 函数的第 4 个位置
    block_seq_tensor = torch.tensor(block_seq, dtype=torch.int32, device='cpu') # 放在 CPU 即可，run 里取 .item()
    
    return (mid_o, mid_o_lse, b_seqlen, block_seq_tensor)
def generate_mat_mul_inputs(M=1024, K=1024, N=1024, dtype=torch.float16, device='cuda', **kwargs):
    # Depending on the PyTorch version, directly initializing float8_e4m3fn with randn might not be supported.
    # The safest way is to generate float32 and cast.
    a = torch.randn((M, K), dtype=torch.float32, device=device).to(dtype)
    b = torch.randn((K, N), dtype=torch.float32, device=device).to(dtype)
    return (a, b)
GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "sin": generate_sin_inputs,
    "rope": generate_rope_inputs,
    "flash_attention": generate_flash_attn_inputs,
    "softmax": generate_softmax_inputs,
    "flash_decode": generate_flash_decode_stage2_inputs,
    "matmul_fp16_fp8": generate_mat_mul_inputs,
}

def get_generator(operator_name):
    if operator_name not in GENERATORS:
        raise ValueError(f"No input generator registered for operator: {operator_name}")
    return GENERATORS[operator_name]
