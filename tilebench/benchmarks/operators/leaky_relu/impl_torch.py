import torch
import torch.nn.functional as F


def _view_rows(n: int, lo: int = 128, hi: int = 8192) -> int | None:
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
        rows = _view_rows(N)
        if rows is not None:
            return F.leaky_relu(input.view(rows, -1), negative_slope=0.01).view(-1)
    return F.leaky_relu(input, negative_slope=0.01)
