import torch

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def run(q, cos, sin):
    # q: [batch, seq_len, n_heads, head_dim]
    # cos, sin: [seq_len, head_dim // 2]
    
    cos = cos.unsqueeze(0).unsqueeze(2) # [1, seq_len, 1, head_dim/2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    head_dim = q.shape[-1]
    q1 = q[..., :head_dim//2]
    q2 = q[..., head_dim//2:]

    q1_out = (q1 * cos) - (q2 * sin)
    q2_out = (q2 * cos) + (q1 * sin)

    return torch.cat((q1_out, q2_out), dim=-1)