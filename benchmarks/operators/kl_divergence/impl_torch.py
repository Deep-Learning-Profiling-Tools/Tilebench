import torch


def run(log_y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    return (y_true * (torch.log(y_true) - log_y_pred)).sum(dim=-1)
