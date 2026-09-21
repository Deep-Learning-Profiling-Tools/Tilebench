import torch
import torch.nn.functional as F


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    return F.conv1d(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        groups=groups,
    )
