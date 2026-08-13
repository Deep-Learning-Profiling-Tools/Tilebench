import torch


torch.backends.cuda.matmul.allow_tf32 = True


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A_3d = A.view(BATCH, M, K)
    B_3d = B.view(BATCH, K, N)
    C = torch.matmul(A_3d, B_3d)
    return C.view(-1)
