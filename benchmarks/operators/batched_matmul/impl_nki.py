"""NKI batched matmul: BATCH independent (M,K)@(K,N) matmuls.

Reuses the existing, already-verified tiled Tensor-Engine matmul from
matmul_fp32_fp16_fp8 (nc_matmul + PSUM K-accumulation) once per batch slice.
@nki.jit caches the compiled kernel by argument shape, so with BATCH slices
sharing one (Mp, Kp, Np) padded shape this compiles once and dispatches
BATCH times. M/N/K here are not generally multiples of
(TILE_M, TILE_K, TILE_N) = (128, 128, 512) (batched_matmul sweeps M=N=K in
steps of 32), so inputs are zero-padded on the host and the output cropped
back -- the same tail strategy matrix_transpose and vector_add use.
"""
import torch

from benchmarks.operators.matmul_fp32_fp16_fp8.impl_nki import (
    TILE_M, TILE_K, TILE_N, run as _matmul_run,
)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int,
        block_size: int = None, autotune: bool = False, **kwargs) -> torch.Tensor:
    A3 = A.view(BATCH, M, K)
    B3 = B.view(BATCH, K, N)

    Mp = ((M + TILE_M - 1) // TILE_M) * TILE_M
    Kp = ((K + TILE_K - 1) // TILE_K) * TILE_K
    Np = ((N + TILE_N - 1) // TILE_N) * TILE_N

    if Mp > M or Kp > K:
        A3 = torch.nn.functional.pad(A3, (0, Kp - K, 0, Mp - M))
    if Kp > K or Np > N:
        B3 = torch.nn.functional.pad(B3, (0, Np - N, 0, Kp - K))

    outputs = [_matmul_run(A3[i], B3[i])[:M, :N] for i in range(BATCH)]
    return torch.stack(outputs, dim=0).reshape(-1)


def get_last_config() -> dict | None:
    return None
