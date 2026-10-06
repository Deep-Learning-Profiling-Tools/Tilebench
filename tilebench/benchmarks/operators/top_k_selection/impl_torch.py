import torch


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    return torch.topk(input.contiguous(), k, largest=True, sorted=True).values
