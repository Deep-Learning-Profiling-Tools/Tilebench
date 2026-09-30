import torch

torch.backends.cuda.matmul.allow_tf32 = True


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    return torch.matmul(a, b)
