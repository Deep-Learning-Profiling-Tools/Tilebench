import torch
import torch.nn.functional as F


def run(x, y):
    if x.device.type == "xla" and x.dtype == torch.float32:
        return x * torch.sigmoid(x) * y
    return F.silu(x) * y
