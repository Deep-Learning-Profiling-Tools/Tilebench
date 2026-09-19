import torch


def run(x, x_keep, p):
    return torch.where(x_keep.bool(), x / (1 - p), torch.zeros_like(x))
