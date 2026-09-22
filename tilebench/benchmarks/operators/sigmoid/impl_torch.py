import torch
import torch.nn.functional as F


def _view_cols(n: int, lo: int = 128, hi: int = 8192) -> int | None:
    for c in range(hi, lo - 1, -1):
        if n % c == 0:
            return c
    return None


def run(X: torch.Tensor, N: int, **kwargs):
    if X.device.type == "xla" and X.dim() == 1 and N % 128 != 0:
        cols = _view_cols(N)
        if cols is not None:
            x2 = X.view(-1, cols)
            pad = (-cols) % 128
            if pad:
                return torch.sigmoid(F.pad(x2, (0, pad)))[:, :cols].reshape(-1)
            return torch.sigmoid(x2).reshape(-1)
    return torch.sigmoid(X)
