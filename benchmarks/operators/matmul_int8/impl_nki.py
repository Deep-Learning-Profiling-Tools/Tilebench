import torch

from benchmarks.operators.matmul_fp32_fp16_fp8.impl_nki import (
    TILE_M, TILE_K, TILE_N, run as _matmul_run,
)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    a_int8 = a.to(torch.int8)
    M, K = a.shape
    K_b, N = b.shape
    assert K == 4 * K_b, "A's K dim must be 4x B's K_b dim"

    parts = []
    for i in range(4):
        mask = 3 << (2 * i)
        b_val = ((b.to(torch.int32) & mask) >> (2 * i)).to(torch.int8) - 1
        parts.append(b_val)
    b_unpacked = torch.cat(parts, dim=0) 

    a_fp32 = a_int8.to(torch.float32)
    b_fp32 = b_unpacked.to(torch.float32)

    Mp = ((M + TILE_M - 1) // TILE_M) * TILE_M
    Kp = ((K + TILE_K - 1) // TILE_K) * TILE_K
    Np = ((N + TILE_N - 1) // TILE_N) * TILE_N

    if Mp > M or Kp > K:
        a_fp32 = torch.nn.functional.pad(a_fp32, (0, Kp - K, 0, Mp - M))
    if Kp > K or Np > N:
        b_fp32 = torch.nn.functional.pad(b_fp32, (0, Np - N, 0, Kp - K))

    out_fp32 = _matmul_run(a_fp32, b_fp32)[:M, :N]
    return out_fp32.round().to(torch.int32)


def get_last_config() -> dict | None:
    return None
