import torch


def run(
    x: torch.Tensor,
    gate: torch.Tensor,
    bias: torch.Tensor,
    block_size: int = 1024,
    **kwargs
):
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")
    return torch.relu(
        x.to(torch.float32) * gate.to(torch.float32) + bias.to(torch.float32)
    )
