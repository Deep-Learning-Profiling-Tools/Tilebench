import torch


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    """
    Reference row-wise argmax using PyTorch.
    Input:  (M, N)
    Output: (M,) int64 indices of the maximum along `dim`
    """
    return torch.argmax(x, dim=dim)
