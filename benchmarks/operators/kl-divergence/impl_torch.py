import torch


def run(p: torch.Tensor, q: torch.Tensor, eps: float, block_size: int = 1024, **kwargs):
    if p.shape != q.shape:
        raise ValueError("Input tensors must have the same shape.")
    p32 = torch.clamp(p.to(torch.float32), min=eps)
    q32 = torch.clamp(q.to(torch.float32), min=eps)
    return p32 * (torch.log(p32) - torch.log(q32))
