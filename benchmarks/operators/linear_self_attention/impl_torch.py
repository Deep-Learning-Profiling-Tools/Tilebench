"""Reference linear self-attention via plain torch matmuls.

  S = phi(K)^T @ V        (D, D)
  Z = sum_m phi(K[m, :])  (D,)
  O = (phi(Q) @ S) / (phi(Q) @ Z + eps)
"""
import torch


def _phi(x: torch.Tensor) -> torch.Tensor:
    # phi(x) = ELU(x) + 1
    # x > 0  -> x + 1
    # x <= 0 -> exp(x)
    return torch.where(x > 0, x + 1.0, torch.exp(x))


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6, **kwargs):
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == K.dtype == V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    phi_q = _phi(Q)
    phi_k = _phi(K)

    # S = phi(K)^T @ V, shape [D, D]
    S = phi_k.transpose(0, 1) @ V

    # Z = sum_m phi(K[m]), shape [D]
    Z = phi_k.sum(dim=0)
    return (phi_q @ S) / ((phi_q @ Z)[:, None] + float(eps))


def get_last_config() -> dict | None:
    return None
