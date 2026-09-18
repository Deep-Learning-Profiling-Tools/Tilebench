"""Reference matmul (Stream-K target). Plain torch.matmul."""
import torch


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    return torch.matmul(a, b)


def get_last_config() -> dict | None:
    return None
