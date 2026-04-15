import torch


def run(data: torch.Tensor, N: int, **kwargs):
    """Reference sort in ascending order. Returns sorted values; input unchanged."""
    return torch.sort(data).values
