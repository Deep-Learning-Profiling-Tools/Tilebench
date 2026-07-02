import torch
import torch.nn.functional as F


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    """
    Reference Conv3d forward pass using PyTorch (cuDNN).
    Input:  (batch, in_channels, D, H, W)
    Weight: (out_channels, in_channels // groups, kD, kH, kW)
    Output: (batch, out_channels, out_D, out_H, out_W)

    Runs in the native dtype — the honest baseline: fp16 uses fp16 Tensor
    Cores, fp32 uses TF32 Tensor Cores (torch.backends.cudnn.allow_tf32
    defaults to True), matching the Triton/cuTile kernels' precision paths.
    """
    return F.conv3d(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        groups=groups,
    )
