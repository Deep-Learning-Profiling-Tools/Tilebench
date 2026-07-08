import torch


def run(X: torch.Tensor, N: int, **kwargs):
    """Element-wise sigmoid: y = 1 / (1 + exp(-x)).

    Native dtype end to end — ATen's unary kernels compute per element at
    higher internal precision, so the old .float() round-trip only added
    two full-size materialisation passes.
    """
    return torch.sigmoid(X)
