import torch
import torch.nn.functional as F


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    """
    Batch Normalization forward pass (training-mode statistics, per-channel).
    Computes mean/var over the batch dim (N), then normalizes and scales.
    """
    mean = torch.mean(input.float(), dim=0)
    var = torch.var(input.float(), dim=0, correction=0)
    result = F.batch_norm(
        input.float(),
        running_mean=mean, running_var=var,
        weight=gamma.float(), bias=beta.float(),
        training=False, momentum=0.0, eps=eps,
    )
    return result.to(input.dtype)
