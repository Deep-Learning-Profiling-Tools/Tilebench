import torch

from benchmarks.operators.bitonic_sort.impl_nki import _bitonic_sort_1d


def run(input: torch.Tensor, N: int, k: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    sorted_ascending = _bitonic_sort_1d(input)
    return torch.flip(sorted_ascending[N - k:], dims=(0,))


def get_last_config() -> dict | None:
    return None
