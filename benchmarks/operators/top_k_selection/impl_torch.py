import torch


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    if input.device.type == "xla":
        return torch.topk(input.contiguous(), k, largest=True, sorted=True)[0]
    return torch.topk(input.contiguous(), k, largest=True, sorted=True).values
