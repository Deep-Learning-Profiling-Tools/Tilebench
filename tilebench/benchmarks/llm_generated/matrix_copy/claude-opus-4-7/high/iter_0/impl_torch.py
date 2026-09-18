import torch


def run(A: torch.Tensor, N: int, **kwargs):
    """
    Matrix copy: B = A, where A is an N x N matrix.
    Uses A + 0 to force a CUDA kernel launch (pure memcpy is DMA,
    invisible to Proton profiler).
    """
    B = A + 0
    return B
