import torch


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    """Reference histogram via torch.bincount.

    bincount only accepts int64 input, so cast first; cast the result back
    to int32 to match the Triton/cuTile output dtype.
    """
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    return torch.bincount(input.to(torch.int64), minlength=num_bins).to(torch.int32)
