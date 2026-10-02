import torch


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if x.device.type == "xla" and x.dtype != torch.float32:
        return x.to(torch.float32).mean(dim=dim)
    return x.mean(dim=dim, dtype=torch.float32)
