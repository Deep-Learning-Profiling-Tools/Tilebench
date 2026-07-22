import torch
import torch.nn.functional as F


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    """
    Batch Normalization forward pass (training-mode statistics, per-channel).

    training=True with no running stats makes F.batch_norm compute the batch
    mean/var itself in ATen's fused single-pass statistics kernel (fp32
    accumulation, biased variance — the same semantics as the DSL kernels),
    instead of a hand-rolled two-reduction version that read the input twice.
    The input stays in its original dtype throughout (mixed-precision path);
    the per-channel gamma/beta casts are (C,)-sized and negligible.
    """
    return F.batch_norm(
        input,
        running_mean=None, running_var=None,
        weight=gamma.float(), bias=beta.float(),
        training=True, momentum=0.0, eps=eps,
    )
