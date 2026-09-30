import torch

from benchmarks.operators.matmul_fp32_fp16_fp8.impl_nki import (
    TILE_M, TILE_K, TILE_N, run as _matmul_run,
    get_last_config as _matmul_get_last_config,
)

# NKI has no faithful Stream-K here -- this intentionally stays a plain-matmul
# stand-in rather than a mislabeled approximation. Stream-K's two defining
# mechanisms are both blocked on this hardware/SDK (checked against the
# installed `nki` package, not a guess):
#   1. Persistent grid sized to the SM/worker count: NKI's launch degree is
#      pinned to the physical LNC (2 on trn2), not an independently-choosable
#      `NUM_SMS`-style worker count the way cutile/triton launch it.
#   2. Atomic accumulation across workers into a shared output tile (cutile's
#      `first_wave`, triton's equivalent): the closest NKI primitive,
#      `nisa.dma_compute(reduce_op=nl.add, ...)`, requires `unique_indices=True`
#      and documents concurrent non-unique-index accumulation as unsupported.
#      The only cross-core sync primitive, `nisa.core_barrier`, is documented
#      for disjoint writes, not combining overlapping partial sums.
# A fixed 2-way split-K kernel is buildable instead, but that's a different
# algorithm (classic split-K, which Stream-K was designed to replace) at a
# worker count (2) too small for load-balancing to matter -- so it isn't
# implemented here either. See git history / PR description for the full
# investigation.


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

    out = _matmul_run(a, b, autotune=autotune)
    return out[:M, :N]


def get_last_config() -> dict | None:
    return _matmul_get_last_config()
