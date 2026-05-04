import torch

_LAST_CONFIG = None


def _phi(x: torch.Tensor) -> torch.Tensor:
    # phi(x) = ELU(x) + 1
    # x > 0  -> x + 1
    # x <= 0 -> exp(x)
    return torch.where(x > 0, x + 1.0, torch.exp(x))


def run(
    Q,
    K,
    V,
    eps: float = 1e-6,
    BLOCK_M: int = 32,
    BLOCK_D: int = 16,
    block_size: int = None,
    autotune: bool = False,
    **kwargs,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_M = int(block_size)

    assert Q.ndim == 2 and K.ndim == 2 and V.ndim == 2
    assert Q.shape == K.shape == V.shape
    assert Q.dtype == torch.float32
    assert K.dtype == torch.float32
    assert V.dtype == torch.float32

    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    phi_q = _phi(Q)
    phi_k = _phi(K)

    # S = phi(K)^T @ V, shape [D, D]
    S = phi_k.transpose(0, 1) @ V

    # Z = sum_m phi(K[m]), shape [D]
    Z = phi_k.sum(dim=0)

    # O = phi(Q) @ S / (phi(Q) @ Z + eps)
    numer = phi_q @ S
    denom = phi_q @ Z
    O = numer / (denom[:, None] + float(eps))

    _LAST_CONFIG = {
        "BLOCK_M": int(BLOCK_M),
        "BLOCK_D": int(BLOCK_D),
        "KV_BLOCK_M": int(BLOCK_M),
        "eps": float(eps),
        "autotune": False,
        "kernel_style": "torch_reference",
    }
    return O


def get_last_config() -> dict | None:
    return _LAST_CONFIG