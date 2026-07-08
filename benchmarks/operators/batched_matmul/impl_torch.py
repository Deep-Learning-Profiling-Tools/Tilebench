import torch

# fp32 matmul uses TF32 tensor cores, matching both DSL backends (Triton
# tl.dot(input_precision="tf32"), cuTile ct.tfloat32 cast before ct.mma)
# and the suite's roofline, which rates fp32 at the TF32 TC peak.
torch.backends.cuda.matmul.allow_tf32 = True


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    """
    Batched matrix multiplication reference.
    A: flat tensor of size BATCH * M * K (viewed as (BATCH, M, K))
    B: flat tensor of size BATCH * K * N (viewed as (BATCH, K, N))
    output: flat tensor of size BATCH * M * N
    Native dtype end to end — cuBLAS accumulates fp16/bf16 GEMMs in fp32
    internally, so no materialized fp32 copies are needed.
    """
    A_3d = A.view(BATCH, M, K)
    B_3d = B.view(BATCH, K, N)
    C = torch.matmul(A_3d, B_3d)
    return C.view(-1)
