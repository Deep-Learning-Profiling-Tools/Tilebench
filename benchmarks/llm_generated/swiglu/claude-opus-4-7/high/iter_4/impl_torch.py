import torch.nn.functional as F


def run(x, y):
    return F.silu(x) * y
