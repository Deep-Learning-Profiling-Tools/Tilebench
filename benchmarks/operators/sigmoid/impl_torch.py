import torch


def run(X: torch.Tensor, N: int, **kwargs):
    """Element-wise sigmoid: y = 1 / (1 + exp(-x))."""
    return torch.sigmoid(X.float()).to(X.dtype)
