import torch
import torch.nn.functional as F


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    x = input.float().reshape(1, 1, -1)        # (N=1, C=1, L)
    w = kernel.float().reshape(1, 1, -1)        # (C_out=1, C_in=1, K)
    result = F.conv1d(x, w).reshape(-1)
    return result.to(input.dtype)
