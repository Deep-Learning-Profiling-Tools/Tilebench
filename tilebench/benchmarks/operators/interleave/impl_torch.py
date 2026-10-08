import torch


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    output[0::2] = A
    output[1::2] = B
    return output
