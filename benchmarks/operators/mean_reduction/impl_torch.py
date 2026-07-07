import torch


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    """
    Reference row-wise mean reduction using PyTorch.
    Accumulates in float32 (dtype=) without materializing an fp32 copy,
    consistent with the Triton/cuTile backends.
    Input:  (M, N)
    Output: (M,) float32
    """
    return x.mean(dim=dim, dtype=torch.float32)
