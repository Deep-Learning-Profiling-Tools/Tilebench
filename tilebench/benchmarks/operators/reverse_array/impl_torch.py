import torch

_XLA_GATHER_MAX = 8_000_000


def run(input: torch.Tensor, N: int, **kwargs):
    if input.device.type == "xla" and N <= _XLA_GATHER_MAX:
        idx = torch.arange(N - 1, -1, -1, device=input.device)
        return input.reshape(-1)[idx]
    return input.flip(0).contiguous()
