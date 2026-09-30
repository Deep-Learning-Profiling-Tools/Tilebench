import torch


def run(X: torch.Tensor, N: int, **kwargs):
    return torch.sigmoid(X)
