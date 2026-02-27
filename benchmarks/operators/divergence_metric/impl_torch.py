import torch


def run(x: torch.Tensor, y: torch.Tensor, eps: float, block_size: int = 1024, **kwargs):
    x32 = x.to(torch.float32)
    y32 = y.to(torch.float32)
    diff = x32 - y32
    return (diff * diff) / (y32 * y32 + eps)
