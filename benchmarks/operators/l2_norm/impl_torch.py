import torch


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    """
    Reference L2 normalisation (per last dimension).

    out = x / sqrt(sum(x^2) + eps) — the same eps-inside-sqrt semantics as
    the Triton/cuTile kernels (F.normalize clamps instead: x / max(||x||,
    eps)). vector_norm(dtype=float32) accumulates in fp32 on the load path
    without materialising an fp32 copy of x.
    """
    norm_sq = torch.linalg.vector_norm(
        x, 2, dim=-1, keepdim=True, dtype=torch.float32).square()
    return (x * torch.rsqrt(norm_sq + eps)).to(x.dtype)
