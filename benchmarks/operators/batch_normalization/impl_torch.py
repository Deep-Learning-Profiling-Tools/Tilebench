import torch
import torch.nn.functional as F


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    return F.batch_norm(
        input,
        running_mean=None, running_var=None,
        weight=gamma.float(), bias=beta.float(),
        training=True, momentum=0.0, eps=eps,
    )
