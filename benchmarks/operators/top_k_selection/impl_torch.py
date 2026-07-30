import torch


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    """Reference top-k via torch.topk (descending, sorted)."""
    return torch.topk(input.contiguous(), k, largest=True, sorted=True).values
