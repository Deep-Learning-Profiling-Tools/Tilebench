import math

import torch


def _histogram_xla(x: torch.Tensor, num_bins: int, chunk: int = 1 << 21) -> torch.Tensor:
    """On-device histogram for Neuron/XLA.

    ``torch.bincount`` / ``torch.histc`` have no XLA lowering (they silently run on
    the host), ``scatter_add_`` / ``index_add_`` lose updates on duplicate indices,
    and ``sort`` is unsupported on trn2 -- so the bin id is split into two digits
    ``v = hi * n_lo + lo`` and the counts are the matmul of the two one-hot codes
    (``counts[hi, lo] = sum_i OH_hi[i, hi] * OH_lo[i, lo]``, exact in fp32
    accumulation), computed in chunks so the one-hot tensors stay small.
    """
    n_lo = 1
    while n_lo * 2 <= math.isqrt(num_bins) and num_bins % (n_lo * 2) == 0:
        n_lo *= 2
    n_hi = num_bins // n_lo
    lo_bits = n_lo.bit_length() - 1
    ar_hi = torch.arange(n_hi, device=x.device, dtype=torch.int32).view(1, n_hi)
    ar_lo = torch.arange(n_lo, device=x.device, dtype=torch.int32).view(1, n_lo)
    acc = torch.zeros(n_hi, n_lo, dtype=torch.float32, device=x.device)
    for i in range(0, x.numel(), chunk):
        xc = x[i:i + chunk]
        # values outside [0, num_bins) match no one-hot column and are dropped
        hi = (xc >> lo_bits).view(-1, 1)
        lo = (xc & (n_lo - 1)).view(-1, 1)
        oh_hi = (hi == ar_hi).to(torch.bfloat16)
        oh_lo = (lo == ar_lo).to(torch.bfloat16)
        acc = acc + (oh_hi.t() @ oh_lo).to(torch.float32)
    return acc.reshape(-1).to(torch.int32)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    if input.device.type == "xla":
        return _histogram_xla(input, num_bins)
    return torch.bincount(input.to(torch.int64), minlength=num_bins).to(torch.int32)
