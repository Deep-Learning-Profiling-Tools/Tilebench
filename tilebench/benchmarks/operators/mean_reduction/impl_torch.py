import torch


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    return x.mean(dim=dim, dtype=torch.float32)
