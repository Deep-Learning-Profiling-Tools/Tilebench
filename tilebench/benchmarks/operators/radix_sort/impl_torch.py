import torch


def run(input: torch.Tensor, N: int, **kwargs):
    return torch.sort(input).values
