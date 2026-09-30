import torch


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    return torch.argmax(x, dim=dim)
