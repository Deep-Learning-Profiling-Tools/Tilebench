import torch
import torch.nn.functional as F


def _view_cols(n: int, lo: int = 128, hi: int = 8192) -> int | None:
    for c in range(hi, lo - 1, -1):
        if n % c == 0:
            return c
    return None


def _interleave(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    output = torch.empty(2 * n, dtype=a.dtype, device=a.device)
    output[0::2] = a
    output[1::2] = b
    return output


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    if A.device.type == "xla" and N % 128 != 0:
        cols = _view_cols(N)
        if cols is not None:
            rows = N // cols
            pad = (-cols) % 128
            a2 = F.pad(A.reshape(rows, cols), (0, pad)).reshape(-1)
            b2 = F.pad(B.reshape(rows, cols), (0, pad)).reshape(-1)
            out = _interleave(a2, b2, a2.numel()).reshape(rows, 2 * (cols + pad))
            return out[:, :2 * cols].reshape(-1)
    return _interleave(A, B, N)
