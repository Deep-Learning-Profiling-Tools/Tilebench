from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

try:
    import cuda.tile_experimental as ct_experimental
except ImportError:  # pragma: no cover
    ct_experimental = None

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=2)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [256, 512, 1024, 2048]
    for occ in [1, 2, 4]
]
_last_autotune_config = None


def _next_pow2(n: int) -> int:
    return 1 << ((n - 1).bit_length()) if n > 1 else 1


@ct.kernel
def _pad_kernel(data_ptr, work_ptr, N, M, TILE: ConstInt):
    """Copy data[0:N] → work[0:N], fill work[N:M] with +inf (mirrors Triton's pad_kernel)."""
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    # ct.gather returns padding_value for OOB indices (>= N or negative).
    vals = ct.gather(data_ptr, offs, padding_value=float("inf"))
    # Tile-aligned store; silently drops the OOB tail (offs >= M).
    ct.store(work_ptr, index=(bid,), tile=vals)


@ct.kernel
def _bitonic_step_kernel(work_ptr, k, j, M, TILE: ConstInt):
    """
    One compare-exchange pass of bitonic sort — cuTile mirror of Triton's method:
      - Each CTA handles TILE positions starting at bid*TILE.
      - For each element at offs, its partner is ixj = offs ^ j.
      - The ixj > offs guard ensures each (offs, ixj) pair is handled by one thread.
      - Read both endpoints, conditionally swap based on ascending/descending sub-sequence,
        write back only from active threads.

    Since cuTile's ct.scatter has no mask parameter, inactive threads route both their
    writes to index M (OOB for a work buffer of size M) which is silently dropped.
    This reproduces Triton's `tl.store(..., mask=active)` semantics without racing on
    the no-op-write pattern.
    """
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    ixj = offs ^ j

    active = (ixj > offs) & (ixj < M) & (offs < M)

    a = ct.gather(work_ptr, offs, padding_value=0.0)
    b = ct.gather(work_ptr, ixj, padding_value=0.0)

    ascending = (offs & k) == 0
    swap = ct.where(ascending, a > b, a < b)
    new_a = ct.where(swap, b, a)
    new_b = ct.where(swap, a, b)

    # Route inactive writes to OOB (index M) so they're silently dropped,
    # avoiding races with active-thread writes to the same positions.
    offs_safe = ct.where(active, offs, M)
    ixj_safe = ct.where(active, ixj, M)

    ct.scatter(work_ptr, offs_safe, new_a)
    ct.scatter(work_ptr, ixj_safe, new_b)


def run(data: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile bitonic sort — direct mirror of the Triton multi-launch approach:
      1. Pad phase: one cuTile kernel copies data[0:N] → work with +inf padding.
      2. Bitonic phase: host loop over (k, j) launches one compare-exchange kernel per step.
    """
    global _last_autotune_config

    if N <= 1:
        return data.clone()

    M = _next_pow2(N)
    work = torch.empty((M,), device=data.device, dtype=data.dtype)
    stream = torch.cuda.current_stream()

    # Pad phase — always uses default config (single launch).
    default_tile = _DEFAULT_CONFIG.tile
    pad_grid = (ct.cdiv(M, default_tile), 1, 1)
    ct.launch(stream, pad_grid, _pad_kernel, (data, work, N, M, default_tile))

    # Bitonic phase — many launches; autotune_launch caches per in-session key.
    if autotune and ct_experimental is not None:
        k = 2
        while k <= M:
            j = k // 2
            while j > 0:
                result = ct_experimental.autotune_launch(
                    stream,
                    grid_fn=lambda cfg: (ct.cdiv(M, cfg.tile), 1, 1),
                    kernel=_bitonic_step_kernel,
                    args_fn=lambda cfg, k=k, j=j: (work, k, j, M, cfg.tile),
                    hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
                    search_space=_SEARCH_SPACE,
                )
                _last_autotune_config = {
                    "tile":      result.tuned_config.tile,
                    "occupancy": result.tuned_config.occupancy,
                }
                j //= 2
            k *= 2
    else:
        cfg = _DEFAULT_CONFIG
        grid = (ct.cdiv(M, cfg.tile), 1, 1)
        k = 2
        while k <= M:
            j = k // 2
            while j > 0:
                ct.launch(stream, grid, _bitonic_step_kernel,
                          (work, k, j, M, cfg.tile))
                j //= 2
            k *= 2

    return work[:N].contiguous()


def get_last_config() -> dict | None:
    return _last_autotune_config
