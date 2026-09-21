import torch
import torch.nn.functional as F


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    x = input.view(N, C, H, W)
    y = F.max_pool2d(x, kernel_size, stride=stride, padding=padding)
    return y.reshape(-1)
