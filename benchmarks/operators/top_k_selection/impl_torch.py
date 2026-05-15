import torch


def run(input: torch.Tensor, N: int, k: int, **kwargs):
    """Reference top-k via torch.topk (descending, sorted)."""
    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N
    return torch.topk(input.contiguous(), k, largest=True, sorted=True).values


def get_last_config() -> dict | None:
    return None
