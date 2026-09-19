import torch


def run(input: torch.Tensor, N: int, **kwargs):
    """
    Reference sort in ascending order (unsigned semantics).
    Sort the int32 tensor directly (no float cast) to avoid fp32 precision
    loss for values above 2^24 — torch.sort on non-negative int32 gives the
    same ordering as unsigned int32 sort, which is what the radix kernel
    produces. Input is never mutated.
    """
    return torch.sort(input).values
