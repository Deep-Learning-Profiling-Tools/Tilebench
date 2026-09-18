import torch


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    """
    Reference row-wise mean reduction using PyTorch.
    Computes in float32 for consistency with the Triton/cuTile backends.
    Input:  (M, N)
    Output: (M,) float32
    """
    return x.float().mean(dim=dim)
