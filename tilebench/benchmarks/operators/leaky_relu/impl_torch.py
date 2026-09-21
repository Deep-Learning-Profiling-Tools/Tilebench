import torch
import torch.nn.functional as F


def _view_rows(n: int, lo: int = 128, hi: int = 8192) -> int | None:
    """Largest divisor of ``n`` in ``[lo, hi]`` (preferring multiples of 128), or None."""
    fallback = None
    for r in range(hi, lo - 1, -1):
        if n % r == 0:
            if r % 128 == 0:
                return r
            if fallback is None:
                fallback = r
    return fallback


def run(input: torch.Tensor, N: int, **kwargs):
    if input.device.type == "xla" and input.dim() == 1 and N % 128 != 0:
        # Neuron/XLA: a 1-D elementwise op on a length that is not a multiple of 128
        # is lowered to ~512-element chunks (~35x slower than the bandwidth bound).
        # The same op on a 2-D view of the tensor compiles to one streaming pass
        # (measured: 2.5M fp32 2.84 ms -> 0.05 ms).
        rows = _view_rows(N)
        if rows is not None:
            return F.leaky_relu(input.view(rows, -1), negative_slope=0.01).view(-1)
    return F.leaky_relu(input, negative_slope=0.01)
