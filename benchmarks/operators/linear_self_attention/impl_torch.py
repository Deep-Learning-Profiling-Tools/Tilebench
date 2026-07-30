import torch


def _phi(x: torch.Tensor) -> torch.Tensor:


    return torch.where(x > 0, x + 1.0, torch.exp(x))


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6, **kwargs):

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    phi_q = _phi(Q)
    phi_k = _phi(K)


    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:

        S = phi_k.transpose(0, 1) @ V


        Z = phi_k.sum(dim=0)
        return (phi_q @ S) / ((phi_q @ Z)[:, None] + float(eps))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32
