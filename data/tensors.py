import torch

def generate_vector_add_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    y = torch.randn(n, dtype=dtype, device=device)
    return (x, y)


def generate_sin_inputs(n, dtype=torch.float32, device='cuda'):
    x = torch.randn(n, dtype=dtype, device=device)
    return (x,)
def generate_rope_inputs(batch_size, seq_len, n_heads, head_dim, dtype=torch.float32, device='cuda', **kwargs):
    """
    根据显式的维度配置生成 RoPE 输入。
    接受 **kwargs 是为了容错 (以防 config 里有 extra params)
    """
    # 构造 Query: [Batch, Seq, Heads, Dim]
    q = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    
    # 构造 Cos/Sin 表: [Seq, Dim // 2]
    # 注意: RoPE 的 Cos/Sin 通常是预计算好的，长度覆盖最大序列长度，且最后一维是 HeadDim/2
    # 这里我们生成刚好够用的长度
    half_dim = head_dim // 2
    cos = torch.randn(seq_len, half_dim, dtype=dtype, device=device)
    sin = torch.randn(seq_len, half_dim, dtype=dtype, device=device)
    
    return (q, cos, sin)

# 在 GENERATORS 字典中添加:
# "rope": generate_rope_inputs

# Registry for input generators
GENERATORS = {
    "vector_add": generate_vector_add_inputs,
    "sin": generate_sin_inputs,
    "rope": generate_rope_inputs,
}

def get_generator(operator_name):
    if operator_name not in GENERATORS:
        raise ValueError(f"No input generator registered for operator: {operator_name}")
    return GENERATORS[operator_name]
