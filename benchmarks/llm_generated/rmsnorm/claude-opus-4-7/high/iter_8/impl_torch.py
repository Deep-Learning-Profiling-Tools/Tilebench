import torch


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm reference implementation.

    out = x / rms(x) * rms_w,  where rms(x) = sqrt(mean(x^2) + eps)

    Computation is done in float32 for numerical stability; the result is
    cast back to the input dtype before returning.

    Args:
        x:     Input tensor of shape (batch, M, K).
        rms_w: Per-channel scale weight of shape (K,).
        eps:   Epsilon for numerical stability (default 1e-6).
    """
    orig_dtype = x.dtype
    x_fp32 = x.float()
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)   # (batch, M, 1)
    rstd = torch.rsqrt(var + eps)                     # (batch, M, 1)
    return (x_fp32 * rstd * rms_w.float()).to(orig_dtype)
