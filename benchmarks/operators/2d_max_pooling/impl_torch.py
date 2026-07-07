import torch
import torch.nn.functional as F


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    """
    2D max pooling reference using F.max_pool2d.
    input:  flat 1D tensor of size N * C * H * W
    output: flat 1D tensor of size N * C * H_out * W_out
            where H_out = (H + 2*padding - kernel_size) // stride + 1
    Max is a pure selection — computed in the input dtype, no upcast needed.
    """
    x = input.view(N, C, H, W)
    y = F.max_pool2d(x, kernel_size, stride=stride, padding=padding)
    return y.reshape(-1)
