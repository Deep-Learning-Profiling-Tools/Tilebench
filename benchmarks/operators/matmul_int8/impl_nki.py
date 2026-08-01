"""NKI int8 (2-bit packed B) matmul.

nc_matmul on trn2 does not support int8 stationary/moving operands (only
float8_e4m3/e5m2, bfloat16, float16, tfloat32, float32 -- confirmed by
probing the compiler directly). A is bounded to [-64, 64] and unpacked B
values to {-1, 0, 1, 2}, so each product is bounded by 128 and the K-summed
accumulator by 128 * K -- at most ~2.6M for the largest configured K
(20480), safely inside fp32's 24-bit exact-integer range (8,388,608). The
Tensor Engine accumulates in fp32 regardless of input dtype, so running the
unpacked operands through the existing fp32 tiled matmul
(matmul_fp32_fp16_fp8) and rounding the result to int32 reproduces the
integer reference bit-for-bit rather than approximately.

B's 2-bit unpacking (byte -> 4 fields -> {-1,0,1,2}) is elementwise,
data-layout work, not part of the M*N*K contraction impl_torch.py's
flops_expr measures -- it's done with plain tensor ops (mirroring
impl_torch.py exactly) rather than a bespoke bit-twiddling NKI kernel.
"""
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
    b_unpacked = torch.cat(parts, dim=0)  # (K, N) int8

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
