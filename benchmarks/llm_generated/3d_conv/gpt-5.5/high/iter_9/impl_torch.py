import torch
import torch.nn.functional as F


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols, **kwargs):
    """
    3D convolution with valid padding (no padding).
    input:  flat 1D tensor of size input_depth * input_rows * input_cols
    kernel: flat 1D tensor of size kernel_depth * kernel_rows * kernel_cols
    output: flat 1D tensor of size output_depth * output_rows * output_cols
    """
    x = input.float().view(1, 1, input_depth, input_rows, input_cols)
    w = kernel.float().view(1, 1, kernel_depth, kernel_rows, kernel_cols)
    y = F.conv3d(x, w)
    return y.view(-1).to(input.dtype)
