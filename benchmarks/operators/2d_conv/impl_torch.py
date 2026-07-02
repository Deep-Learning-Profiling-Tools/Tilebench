import torch
import torch.nn.functional as F


def run(input: torch.Tensor, weight: torch.Tensor,
        stride: int = 1, padding: int = 1, groups: int = 1, **kwargs):
    """
    Reference Conv2d forward pass using PyTorch (cuDNN).
    Input:  (batch, in_channels, H, W)
    Weight: (out_channels, in_channels // groups, kH, kW)
    Output: (batch, out_channels, out_H, out_W)

    Runs in the native dtype — the honest baseline: fp16 uses fp16 Tensor
    Cores, fp32 uses TF32 Tensor Cores (torch.backends.cudnn.allow_tf32
    defaults to True), matching the Triton/cuTile kernels' precision paths.
    The old upcast-to-fp32 version forced every dtype onto the TF32 path.
    """
    return F.conv2d(
        input,
        weight,
        bias=None,
        stride=stride,
        padding=padding,
        groups=groups,
    )
