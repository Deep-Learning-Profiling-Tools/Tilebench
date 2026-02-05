import torch

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def run(q, cos, sin):
    # q: [batch, seq_len, n_heads, head_dim]
    # cos, sin: [seq_len, head_dim // 2]
    
    # 调整 cos/sin 形状以进行广播
    # cos: [seq_len, head_dim/2] -> [1, seq_len, 1, head_dim]
    # 注意：RoPE 是对偶数索引和奇数索引操作，或者前半部分和后半部分。
    # Triton 实现中是前半部分(Q1)和后半部分(Q2)。
    # 公式：Q1_new = Q1*cos - Q2*sin
    #       Q2_new = Q2*cos + Q1*sin
    
    # 为了匹配 Triton 实现的 "前半部分 vs 后半部分" 逻辑：
    # 我们需要将 cos/sin 扩展到 [1, seq_len, 1, head_dim/2]
    cos = cos.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim/2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # 将 Q 分割为前半部分和后半部分
    head_dim = q.shape[-1]
    q1 = q[..., :head_dim//2]
    q2 = q[..., head_dim//2:]
    
    # 应用旋转
    q1_out = (q1 * cos) - (q2 * sin)
    q2_out = (q2 * cos) + (q1 * sin)
    
    # 拼接回原始形状
    return torch.cat((q1_out, q2_out), dim=-1)