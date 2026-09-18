import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference int8 matmul with 2-bit packed B.

    A: (M, K) int8.
    B: (K_b, N) uint8 — each byte holds 4 packed 2-bit fields. Field i
       (i = 0..3) is mask `3 << (2*i)` shifted right by `2*i`, then minus 1
       to map {0, 1, 2, 3} → {-1, 0, 1, 2}.
    Output: (M, N) int32.
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

    out = torch.matmul(a_int8.to(torch.float32), b_unpacked.to(torch.float32))
    return out.to(torch.int32)
