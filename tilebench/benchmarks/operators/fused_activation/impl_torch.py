import torch
import torch.nn.functional as F


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    return F.silu(x * gate + bias)
