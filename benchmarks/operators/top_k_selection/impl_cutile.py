"""cuTile top-k selection — same multi-launch bitonic sort as impl_triton.py.

Each compare-exchange kernel handles TILE pairs per CTA (mirrors Triton's
BLOCK_SIZE), using ct.gather for runtime-strided loads and ct.scatter for
runtime-strided stores. Inactive lanes route their writes to OOB index N
(silently dropped) — equivalent to Triton's `tl.store(..., mask=valid)`.

Tune ONCE per padding_len, not per (stage, stride) — `stage` and `stride`
are runtime ints baked in via args_fn, so the optimal config depends only
on padding_len and TILE. This matches Triton's `key=["N"]` cache.
"""
from types import SimpleNamespace

import cuda.tile as ct
import numpy as np
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_last_autotune_config: dict = {}

# Defaults and search space mirror impl_triton.py 1-to-1:
#   tile       ↔ BLOCK_SIZE        same values
#   occupancy  ↔ num_warps         nw * occ ≈ 64 on B200, so cuTile's
#                                   occ ∈ [8, 16, 32] pairs with Triton's
#                                   nw ∈ [8, 4, 2]. Default occ=16 ↔ nw=4.
_DEFAULT_CONFIG = SimpleNamespace(tile=1024, occupancy=16)
_SEARCH_SPACE = [
    SimpleNamespace(tile=t, occupancy=occ)
    for t in [512, 1024, 2048]
    for occ in [8, 16, 32]
]


@ct.kernel
def _bitonic_step_kernel(
    input_ptr,
    N,
    stage,
    stride,
    TILE: ConstInt,
):
    """One compare-exchange pass over TILE pairs per CTA — mirrors Triton."""
    bid = ct.bid(0)
    offset = bid * TILE + ct.arange(TILE, dtype=np.int32)

    slice_1_offset = (offset // stride) * (2 * stride) + (offset % stride)
    slice_2_offset = slice_1_offset + stride

    valid_1 = slice_1_offset < N
    valid_2 = slice_2_offset < N

    # Two-step OOB handling (defensive):
    #   1. Clamp OOB lanes to a safe in-range index (0). ct.gather's
    #      padding_value is documented to fire for negative / out-of-range
    #      indices only; an in-range (but logically invalid) index returns
    #      the real element at that address. So clamping to 0 is needed
    #      first to avoid undefined-bounds reads.
    #   2. Override the gathered values to -inf for invalid lanes. This is
    #      what actually neutralises them in the bitonic compare-exchange,
    #      regardless of what the gather returned.
    safe_1 = ct.where(valid_1, slice_1_offset, 0)
    safe_2 = ct.where(valid_2, slice_2_offset, 0)
    slice_1_t = ct.gather(input_ptr, safe_1, padding_value=-float("inf"))
    slice_2_t = ct.gather(input_ptr, safe_2, padding_value=-float("inf"))
    slice_1_t = ct.where(valid_1, slice_1_t, -float("inf"))
    slice_2_t = ct.where(valid_2, slice_2_t, -float("inf"))

    descend = ((slice_1_offset // stage) % 2) == 1
    greater = slice_1_t > slice_2_t
    swap = descend == greater

    new_slice_1_t = ct.where(swap, slice_2_t, slice_1_t)
    new_slice_2_t = ct.where(swap, slice_1_t, slice_2_t)

    # Route inactive writes to OOB index N (silently dropped by ct.scatter),
    # equivalent to Triton's mask=valid.
    store_1 = ct.where(valid_1, slice_1_offset, N)
    store_2 = ct.where(valid_2, slice_2_offset, N)
    ct.scatter(input_ptr, store_1, new_slice_1_t)
    ct.scatter(input_ptr, store_2, new_slice_2_t)


# Module-level: caches replace_hints per-occupancy and autotune-best per
# padding_len. Mirrors Triton's @triton.autotune(key=["N"]) — one sweep per
# problem size, reused across all log^2(padding_len)/2 step launches.
_tuner = CutileAutotuner(_bitonic_step_kernel)


def _next_pow2(x: int) -> int:
    return 1 << (int(x) - 1).bit_length()


def run(input: torch.Tensor, N: int, k: int,
        block_size: int = None, autotune: bool = False, **kwargs):

    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.float32
    assert 1 <= k <= N

    input = input.contiguous()
    padding_len = _next_pow2(N)
    input_padding = torch.empty((padding_len,), device=input.device, dtype=input.dtype)
    input_padding[:N] = input
    input_padding[N:] = -float("inf")

    stream = torch.cuda.current_stream()
    pair_count = padding_len // 2

    # Tune ONCE per padding_len. (stage, stride) are runtime ints, so the
    # optimal cfg depends only on padding_len and TILE. Use the first step's
    # (stage, stride) for the tuning launches.
    if autotune:
        stage0, stride0 = 2, 1
        cfg = _tuner.tune_or_cached(
            shape_key=(padding_len,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: ((pair_count + cfg.tile - 1) // cfg.tile, 1, 1),
            args_fn=lambda cfg: (input_padding, padding_len, stage0, stride0, cfg.tile),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"tile": cfg.tile, "occupancy": cfg.occupancy})
    else:
        TILE = int(block_size) if block_size is not None else _DEFAULT_CONFIG.tile
        cfg = SimpleNamespace(tile=TILE, occupancy=_DEFAULT_CONFIG.occupancy)

    kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)
    grid = ((pair_count + cfg.tile - 1) // cfg.tile, 1, 1)

    stage = 2
    while stage <= padding_len:
        stride = stage >> 1
        while stride > 0:
            ct.launch(stream, grid, kernel,
                      (input_padding, padding_len, stage, stride, cfg.tile))
            stride >>= 1
        stage <<= 1

    return input_padding[:k].clone()


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
