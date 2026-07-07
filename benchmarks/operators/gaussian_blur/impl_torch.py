import torch
import torch.nn.functional as F


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    """
    2D Gaussian blur with same-size zero-padded boundary.
    input:  flat 1D tensor of size input_rows * input_cols
    kernel: flat 1D tensor of size kernel_rows * kernel_cols (normalized, sums to 1)
    output: flat 1D tensor of size input_rows * input_cols
    Native dtype end to end — cuDNN accumulates fp16 convolutions in fp32
    internally, so no materialized fp32 copies are needed.
    """
    x = input.view(1, 1, input_rows, input_cols)
    w = kernel.view(1, 1, kernel_rows, kernel_cols)
    y = F.conv2d(x, w, padding=(kernel_rows // 2, kernel_cols // 2))
    y = y[:, :, :input_rows, :input_cols]
    return y.reshape(-1)
