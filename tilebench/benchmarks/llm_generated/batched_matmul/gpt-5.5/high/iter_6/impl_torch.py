import torch


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    """
    Batched matrix multiplication reference.
    A: flat tensor of size BATCH * M * K (viewed as (BATCH, M, K))
    B: flat tensor of size BATCH * K * N (viewed as (BATCH, K, N))
    output: flat tensor of size BATCH * M * N
    """
    A_3d = A.view(BATCH, M, K).float()
    B_3d = B.view(BATCH, K, N).float()
    C = torch.matmul(A_3d, B_3d).to(A.dtype)
    return C.view(-1)
