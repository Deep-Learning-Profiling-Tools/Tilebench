import torch
import torch.nn.functional as F


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    """
    Batch Normalization forward pass (training-mode statistics, per-channel).
    Computes mean/var over the batch dim (N), then normalizes and scales.

    Statistics accumulate in fp32 on the load path (dtype=) with the same
    E[x^2] - E[x]^2 formulation as the DSL kernels; the input itself is never
    materialised in fp32 (the old input.float() calls copied the 80 MB input
    three times and dominated the fp16 timing). F.batch_norm takes the
    original-dtype input with fp32 stats — the standard mixed-precision path —
    so the output is already in the input dtype. The per-channel gamma/beta
    casts are (C,)-sized and negligible.
    """
    mean = torch.mean(input, dim=0, dtype=torch.float32)
    mean_sq = torch.mean(input.square(), dim=0, dtype=torch.float32)
    var = mean_sq - mean.square()
    return F.batch_norm(
        input,
        running_mean=mean, running_var=var,
        weight=gamma.float(), bias=beta.float(),
        training=False, momentum=0.0, eps=eps,
    )
