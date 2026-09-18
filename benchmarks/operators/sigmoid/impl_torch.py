import torch
import torch.nn.functional as F


def _view_cols(n: int, lo: int = 128, hi: int = 8192) -> int | None:
    """Largest divisor of ``n`` in ``[lo, hi]`` (row length of the 2-D view), or None."""
    for c in range(hi, lo - 1, -1):
        if n % c == 0:
            return c
    return None


def run(X: torch.Tensor, N: int, **kwargs):
    if X.device.type == "xla" and X.dim() == 1 and N % 128 != 0:
        # Neuron/XLA: a 1-D activation whose length is not a multiple of 128 is lowered
        # to ~512-element chunks (~30x slower than the bandwidth bound), and so is any
        # 1-D pad/cat around it. On a 2-D view with long rows, padding the row length
        # to a multiple of 128 compiles to strided DMAs plus one streaming activation.
        cols = _view_cols(N)
        if cols is not None:
            x2 = X.view(-1, cols)
            pad = (-cols) % 128
            if pad:
                return torch.sigmoid(F.pad(x2, (0, pad)))[:, :cols].reshape(-1)
            return torch.sigmoid(x2).reshape(-1)
    return torch.sigmoid(X)
