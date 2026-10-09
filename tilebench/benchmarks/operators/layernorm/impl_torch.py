import torch
import torch.nn.functional as F


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if x.device.type == "neuron" and x.dtype in (torch.float16, torch.bfloat16):
        return F.layer_norm(x.float(), (x.shape[-1],), weight.float(), bias.float(), eps).to(x.dtype)
    return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
