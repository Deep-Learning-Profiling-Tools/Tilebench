import torch


def run(input: torch.Tensor, N: int, **kwargs):
    """Element-wise Leaky ReLU: y = x if x > 0 else 0.01 * x."""
    return torch.where(input > 0, input, 0.01 * input)
