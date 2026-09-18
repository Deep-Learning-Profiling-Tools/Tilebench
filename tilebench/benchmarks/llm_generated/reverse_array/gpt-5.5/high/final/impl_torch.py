import torch


def run(input: torch.Tensor, N: int, **kwargs):
    return input.flip(0).contiguous()
