import torch

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def run(q, cos, sin):


    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    head_dim = q.shape[-1]
    q1 = q[..., :head_dim//2]
    q2 = q[..., head_dim//2:]

    q1_out = (q1 * cos) - (q2 * sin)
    q2_out = (q2 * cos) + (q1 * sin)

    return torch.cat((q1_out, q2_out), dim=-1)
