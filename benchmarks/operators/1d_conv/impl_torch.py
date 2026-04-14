import torch


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    windows = input.float().unfold(0, kernel_size, 1)   # [output_size, kernel_size]
    result = (windows * kernel.float()).sum(dim=1)
    return result.to(input.dtype)
