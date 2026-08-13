import torch
import torch.nn.functional as F


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.rms_norm(x, x.shape[-1:], weight=rms_w, eps=eps)
