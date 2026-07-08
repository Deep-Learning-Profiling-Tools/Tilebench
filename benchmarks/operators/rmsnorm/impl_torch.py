import torch
import torch.nn.functional as F


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm reference implementation.

    out = x / rms(x) * rms_w,  where rms(x) = sqrt(mean(x^2) + eps)

    Uses the fused ATen op F.rms_norm (fp32 internal accumulation for
    half dtypes) — the previous hand-written pow/mean/rsqrt/mul chain
    launched ~5 kernels plus a materialised fp32 copy of x and was not a
    representative PyTorch baseline.

    Args:
        x:     Input tensor of shape (batch, M, K).
        rms_w: Per-channel scale weight of shape (K,).
        eps:   Epsilon for numerical stability (default 1e-6).
    """
    return F.rms_norm(x, x.shape[-1:], weight=rms_w, eps=eps)
