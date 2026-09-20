import torch
import torch.nn.functional as F


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    x = input.view(1, 1, input_rows, input_cols)
    w = kernel.view(1, 1, kernel_rows, kernel_cols)
    y = F.conv2d(x, w, padding=(kernel_rows // 2, kernel_cols // 2))
    y = y[:, :, :input_rows, :input_cols]
    return y.reshape(-1)
