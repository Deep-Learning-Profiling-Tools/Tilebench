import torch

from benchmarks.operators.matmul_fp32_fp16_fp8.impl_nki import (
    TILE_M, TILE_K, TILE_N, run as _matmul_run,
)


def run(a: torch.Tensor, b: torch.Tensor, block_size: int = None,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape

    Mp = ((M + TILE_M - 1) // TILE_M) * TILE_M
    Kp = ((K + TILE_K - 1) // TILE_K) * TILE_K
    Np = ((N + TILE_N - 1) // TILE_N) * TILE_N

    if Mp > M or Kp > K:
        a = torch.nn.functional.pad(a, (0, Kp - K, 0, Mp - M))
    if Kp > K or Np > N:
        b = torch.nn.functional.pad(b, (0, Np - N, 0, Kp - K))

    out = _matmul_run(a, b)
    return out[:M, :N]


def get_last_config() -> dict | None:
    return None
