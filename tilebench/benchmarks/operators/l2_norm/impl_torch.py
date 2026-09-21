import torch


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    norm_sq = torch.linalg.vector_norm(
        x, 2, dim=-1, keepdim=True, dtype=torch.float32).square()
    return (x * torch.rsqrt(norm_sq + eps)).to(x.dtype)
