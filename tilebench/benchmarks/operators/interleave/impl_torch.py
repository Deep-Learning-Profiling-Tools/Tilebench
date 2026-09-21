import torch
import torch.nn.functional as F


def _view_cols(n: int, lo: int = 128, hi: int = 8192) -> int | None:
    """Largest divisor of ``n`` in ``[lo, hi]`` (row length of the 2-D view), or None."""
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
        # Neuron/XLA: the strided assignments compile to one streaming pass when the
        # length is a multiple of 128 (2M elements: 0.08 ms) but to element-wise
        # scatters otherwise (1M: 10 ms, 3M: 226 ms). Pad the row length of a 2-D view
        # to a multiple of 128 (strided DMAs), interleave the padded flat tensors and
        # drop the padding again.
        cols = _view_cols(N)
        if cols is not None:
            rows = N // cols
            pad = (-cols) % 128
            a2 = F.pad(A.reshape(rows, cols), (0, pad)).reshape(-1)
            b2 = F.pad(B.reshape(rows, cols), (0, pad)).reshape(-1)
            out = _interleave(a2, b2, a2.numel()).reshape(rows, 2 * (cols + pad))
            return out[:, :2 * cols].reshape(-1)
    return _interleave(A, B, N)
