"""NKI top_k_selection: reuses bitonic_sort's proven ascending compare-exchange
network (see benchmarks/operators/bitonic_sort/impl_nki.py) to fully sort the
input, then takes the last k elements (the largest) and reverses them to the
reference's descending order. A dedicated partial top-k (extract-and-mask
loop, or a truncated bitonic merge) would touch less data, but this is
correct and reuses an already-verified kernel rather than a second bespoke
sorting implementation.
"""
import torch

from benchmarks.operators.bitonic_sort.impl_nki import _bitonic_sort_1d


def run(input: torch.Tensor, N: int, k: int, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    sorted_ascending = _bitonic_sort_1d(input)
    return torch.flip(sorted_ascending[N - k:], dims=(0,))


def get_last_config() -> dict | None:
    return None
