import torch


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    return x.to(torch.float16)
