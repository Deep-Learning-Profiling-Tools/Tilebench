import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference int8 matmul with 2-bit packed B.

    A: (M, K) int8.
    B: (K_b, N) uint8 — each byte holds 4 packed 2-bit fields. Field i
       (i = 0..3) is mask `3 << (2*i)` shifted right by `2*i`, then minus 1
       to map {0, 1, 2, 3} → {-1, 0, 1, 2}.
    Output: (M, N) int32.

    The GEMM runs as an fp32 EMULATION, not integer tensor cores:
    torch.matmul has no int8 CUDA path, and torch._int_mm (the exposed
    int8 GEMM entry) fails with CUBLAS_STATUS_NOT_INITIALIZED for every
    layout/size on this stack (torch 2.10.0+cu130, B200 sm_100 — known
    cublasLt int8/COMPUTE_32I gaps on Blackwell + CUDA 13). With TF32
    disabled the emulation is still an EXACT reference: |acc| <=
    128 * 2 * K <= 5.3M for the swept K, well inside fp32's 2^24
    exactly-representable integer range. Its performance is an
    emulation baseline — flag alongside the fp8 case in the paper.
    """
    a_int8 = a.to(torch.int8)
    M, K = a.shape
    K_b, N = b.shape
    assert K == 4 * K_b, "A's K dim must be 4× B's K_b dim"

    b_unpacked = torch.empty((K, N), dtype=torch.int8, device=a.device)
    for i in range(4):
        mask = 3 << (2 * i)
        b_val = ((b.to(torch.int32) & mask) >> (2 * i)).to(torch.int8) - 1
        b_unpacked[i * K_b : (i + 1) * K_b, :] = b_val

    # Exactness requires full fp32 multiply — guard against a TF32 flag
    # leaking in from another operator module in shared-process tools.
    torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.matmul(a_int8.to(torch.float32), b_unpacked.to(torch.float32))
    return out.to(torch.int32)
