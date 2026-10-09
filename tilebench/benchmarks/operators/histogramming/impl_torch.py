import torch


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    return torch.bincount(input.to(torch.int64), minlength=num_bins).to(torch.int32)
