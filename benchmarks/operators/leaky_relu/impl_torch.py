import torch
import torch.nn.functional as F


def run(input: torch.Tensor, N: int, **kwargs):
    return F.leaky_relu(input, negative_slope=0.01)
