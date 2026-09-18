import torch


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_int8 = a.to(torch.int8)
    M, K = a.shape
    K_b, N = b.shape
    assert K == 4 * K_b, "A's K dim must be 4× B's K_b dim"

    b_unpacked = torch.empty((K, N), dtype=torch.int8, device=a.device)
    for i in range(4):
        mask = 3 << (2 * i)
        b_val = ((b.to(torch.int32) & mask) >> (2 * i)).to(torch.int8) - 1
        b_unpacked[i * K_b : (i + 1) * K_b, :] = b_val


    torch.backends.cuda.matmul.allow_tf32 = False
    out = torch.matmul(a_int8.to(torch.float32), b_unpacked.to(torch.float32))
    return out.to(torch.int32)
