"""Reference KL divergence forward (per-row sum), matching impl_triton/cutile.

Convention (matches PyTorch F.kl_div with log_target=False):
    log_y_pred  is log-probabilities  (output of log_softmax)
    y_true      is plain probabilities (output of softmax)

Per-row formula:
    loss[b] = sum_s y_true[b, s] * (log(y_true[b, s]) - log_y_pred[b, s])

No final batch reduction -- caller can do .mean() / B for batchmean.
"""
import torch


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return (y_true * (torch.log(y_true) - log_y_pred)).sum(dim=-1)


def get_last_config() -> dict | None:
    return None
