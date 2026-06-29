from types import SimpleNamespace

import cuda.tile as ct
import torch

from core.cutile_autotune import CutileAutotuner

ConstInt = ct.Constant[int]

_DEFAULT_CONFIG = SimpleNamespace(occupancy=4)
_SEARCH_SPACE = [SimpleNamespace(occupancy=occ) for occ in [4, 8, 16, 32]]
_last_autotune_config: dict = {}

_BLOCK_SIZE = 1024
_BLOCK_BB = 128  # prefix-sum kernel for the second-layer buffer (matches Triton)


@ct.kernel
def _count_ones_in_block(input_ptr, block_sum_ptr, N, bit, TILE: ConstInt):
    bid = ct.bid(0)
    offset = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    mask = offset < N

    idx_safe = ct.where(mask, offset, -1)
    block = ct.gather(input_ptr, idx_safe, padding_value=0)

    bit_mask = ct.astype((block >> bit) & 1, ct.int32)
    local_sum = ct.sum(bit_mask, axis=0, keepdims=True)  # (1,)
    ct.store(block_sum_ptr, index=(bid,), tile=local_sum)


@ct.kernel
def _count_ones_per_block_blocks(first_sum_ptr, block_block_sum_ptr, K, TILE: ConstInt):
    bid = ct.bid(0)
    offset = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    mask = offset < K

    idx_safe = ct.where(mask, offset, -1)
    vals = ct.gather(first_sum_ptr, idx_safe, padding_value=0)

    local_sum = ct.sum(vals, axis=0, keepdims=True)
    ct.store(block_block_sum_ptr, index=(bid,), tile=local_sum)


@ct.kernel
def _compute_prefix_sums_per_block_of_blocks(block_block_sum_ptr, global_ones_ptr, L, TILE_BB: ConstInt):
    # Grid of 1 process.
    offset = ct.arange(TILE_BB, dtype=ct.int32)
    mask = offset < L

    idx_safe = ct.where(mask, offset, -1)
    vals = ct.gather(block_block_sum_ptr, idx_safe, padding_value=0)

    # Exclusive cumsum.
    cum = ct.cumsum(vals, axis=0)
    excl = cum - vals

    # Masked in-place write: route inactive positions to OOB (index L, beyond the buffer).
    store_idx = ct.where(mask, offset, L)
    ct.scatter(block_block_sum_ptr, store_idx, excl)

    # Total sum → global_ones[0].
    total = ct.sum(vals, axis=0, keepdims=True)
    ct.store(global_ones_ptr, index=(0,), tile=total)


@ct.kernel
def _compute_prefix_sums_per_block(first_sum_ptr, block_block_sum_ptr, K, TILE: ConstInt):
    bid = ct.bid(0)
    offset = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    mask = offset < K

    idx_safe = ct.where(mask, offset, -1)
    sums = ct.gather(first_sum_ptr, idx_safe, padding_value=0)

    # Block-level prefix loaded as scalar.
    prefix = ct.load(block_block_sum_ptr, index=(bid,), shape=())

    cum = ct.cumsum(sums, axis=0)
    excl = cum - sums + prefix

    # Masked in-place write: inactive positions → index K (OOB).
    store_idx = ct.where(mask, offset, K)
    ct.scatter(first_sum_ptr, store_idx, excl)


@ct.kernel
def _radix_sort_kernel(input_ptr, output_ptr, first_sum_ptr, global_ones_ptr,
                       bit, N, TILE: ConstInt):
    """Scatter each element to its correct position based on the current bit."""
    bid = ct.bid(0)
    offset = bid * TILE + ct.arange(TILE, dtype=ct.int32)
    mask = offset < N

    # Scalar prefix sums for this block.
    ones_before = ct.load(first_sum_ptr, index=(bid,), shape=())
    zeros_before = bid * TILE - ones_before

    # Block of input (OOB → 0 via padding; block values of OOB lanes are unused after masking).
    idx_safe = ct.where(mask, offset, -1)
    block = ct.gather(input_ptr, idx_safe, padding_value=0)

    mask_bits = ct.astype((block >> bit) & 1, ct.int32)

    ones_in_block = ct.cumsum(mask_bits, axis=0)
    ones_rank = ones_in_block - mask_bits

    zeros_in_block = ct.cumsum(1 - mask_bits, axis=0)
    zeros_rank = zeros_in_block - (1 - mask_bits)

    # Global zeros count (derived from N and total ones so far).
    global_ones = ct.load(global_ones_ptr, index=(0,), shape=())
    global_zeros = N - global_ones

    offset_values = ct.where(
        mask_bits == 0,
        ct.astype(zeros_before, ct.int32) + zeros_rank,
        ct.astype(global_zeros, ct.int32) + ct.astype(ones_before, ct.int32) + ones_rank,
    )

    # Scatter to destination; inactive lanes route to N (OOB, silently dropped).
    dest = ct.where(mask, offset_values, N)
    ct.scatter(output_ptr, dest, block)


# Module-level: caches replace_hints per-occupancy and autotune-best per shape.
# Mirrors Triton's @triton.autotune(key=["N"]) — one sweep per problem size,
# reused across all 32 bit-pass scatter launches.
_tuner = CutileAutotuner(_radix_sort_kernel)


def run(input: torch.Tensor, N: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    cuTile 32-pass 1-bit radix sort — direct mirror of the Triton pipeline:
      for each bit in 0..31:
        1. per-block count of bit=1
        2. block-of-blocks count + hierarchical prefix sum
        3. per-block prefix sum (adds block-level offset)
        4. scatter to final position
    """

    if N <= 1:
        return input.clone()

    work = input.clone()
    output = torch.empty_like(input)

    K = (N + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    L = (K + _BLOCK_SIZE - 1) // _BLOCK_SIZE

    first_layer = torch.empty((K,), dtype=torch.int32, device=input.device)
    second_layer = torch.empty((L,), dtype=torch.int32, device=input.device)
    # 1-element tensor instead of 0-d for cleaner ct.load/ct.store semantics.
    global_ones = torch.empty((1,), dtype=torch.int32, device=input.device)

    stream = torch.cuda.current_stream()
    grid = (K, 1, 1)
    grid_second = (L, 1, 1)
    grid_third = (1, 1, 1)

    # Tune the scatter ONCE per N (matching Triton's `key=["N"]`); the optimal
    # config doesn't depend on `bit`, only on the problem size.
    if autotune:
        cfg = _tuner.tune_or_cached(
            shape_key=(N,),
            search_space=_SEARCH_SPACE,
            stream=stream,
            grid_fn=lambda cfg: grid,
            args_fn=lambda cfg: (
                work, output, first_layer, global_ones, 0, N, _BLOCK_SIZE,
            ),
            hints_fn=lambda cfg: {"occupancy": cfg.occupancy},
        )
        _last_autotune_config.clear()
        _last_autotune_config.update({"occupancy": cfg.occupancy})
    else:
        cfg = _DEFAULT_CONFIG

    scatter_kernel = _tuner.kernel_with_hints(occupancy=cfg.occupancy)

    for bit in range(32):
        ct.launch(stream, grid, _count_ones_in_block,
                  (work, first_layer, N, bit, _BLOCK_SIZE))
        ct.launch(stream, grid_second, _count_ones_per_block_blocks,
                  (first_layer, second_layer, K, _BLOCK_SIZE))
        ct.launch(stream, grid_third, _compute_prefix_sums_per_block_of_blocks,
                  (second_layer, global_ones, L, _BLOCK_BB))
        ct.launch(stream, grid_second, _compute_prefix_sums_per_block,
                  (first_layer, second_layer, K, _BLOCK_SIZE))
        ct.launch(stream, grid, scatter_kernel,
                  (work, output, first_layer, global_ones, bit, N, _BLOCK_SIZE))

        work.copy_(output)

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
