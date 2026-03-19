import torch


def run(a_q: torch.Tensor, b_q: torch.Tensor, scale: float, block_size: int = 1024, **kwargs):
    if a_q.dim() != 2 or b_q.dim() != 2:
        raise ValueError("matmul-int8 expects 2D inputs.")
    if a_q.shape[1] != b_q.shape[0]:
        raise ValueError("Inner dimensions must match for GEMM.")
    a = a_q.to(torch.float32) * scale
    b = b_q.to(torch.float32) * scale
    return torch.matmul(a, b)
