import torch


def run(input: torch.Tensor, N: int, **kwargs):
    return torch.where(input > 0, input, 0.01 * input)
