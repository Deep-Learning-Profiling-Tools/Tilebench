import torch
import torch.nn.functional as F


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
