import torch
import torch.nn.functional as F


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    """
    Reference Conv2d forward pass using PyTorch.
    Input:  (batch, in_channels, H, W)
    Weight: (out_channels, in_channels // groups, kH, kW)
    Output: (batch, out_channels, out_H, out_W)
    """
    return F.conv2d(
        input.float(),
        weight.float(),
        bias=None,
        stride=stride,
        padding=padding,
        groups=groups,
    ).to(input.dtype)
