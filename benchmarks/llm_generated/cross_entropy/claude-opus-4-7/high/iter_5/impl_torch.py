import torch
import torch.nn.functional as F


def run(logits: torch.Tensor, targets: torch.Tensor):
    if logits.dim() != 2:
        raise ValueError(
            "logits must be a 2D tensor with shape [batch_size, num_classes]."
        )
    if targets.dim() != 1:
        raise ValueError("targets must be a 1D tensor with shape [batch_size].")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("Batch size of logits and targets must match.")
    return F.cross_entropy(logits, targets, reduction="none")
