import torch


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    return x.transpose(0, 1).contiguous()
