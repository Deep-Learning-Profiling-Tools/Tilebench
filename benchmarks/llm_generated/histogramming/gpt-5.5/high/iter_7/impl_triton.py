import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _zero_i32_kernel(ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(ptr + offs, tl.zeros((BLOCK,), dtype=tl.int32), mask=offs < total)


@triton.jit
def _hist_atomic_partition_kernel(
    input_ptr,
    partial_ptr,
    N,
    BINS: tl.constexpr,
    PARTITIONS: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, TILE)
    offs = pid * TILE + lanes
    mask = offs < N

    vals = tl.load(
        input_ptr + offs,
        mask=mask,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    part = pid & (PARTITIONS - 1)
    one = tl.full((TILE,), 1, dtype=tl.int32)

    tl.atomic_add(
        partial_ptr + part * BINS + vals,
        one,
        sem="relaxed",
        scope="gpu",
        mask=mask,
    )


@triton.jit
def _reduce_partials_kernel(
    partial_ptr,
    output_ptr,
    BINS: tl.constexpr,
    PARTITIONS: tl.constexpr,
    BIN_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    bins = bid * BIN_TILE + tl.arange(0, BIN_TILE)
    parts = tl.arange(0, PARTITIONS)

    vals = tl.load(
        partial_ptr + parts[:, None] * BINS + bins[None, :],
        mask=bins[None, :] < BINS,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    sums = tl.sum(vals, axis=0)
    tl.store(output_ptr + bins, sums, mask=bins < BINS)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    TILE = 1024
    PARTITIONS = 64
    ZERO_BLOCK = 1024
    BIN_TILE = 16

    ZERO_WARPS = 4
    HIST_WARPS = 4
    REDUCE_WARPS = 4
    NUM_STAGES = 2

    partial = torch.empty((PARTITIONS, num_bins), device=input.device, dtype=torch.int32)
    total_partial = PARTITIONS * num_bins

    _zero_i32_kernel[(triton.cdiv(total_partial, ZERO_BLOCK),)](
        partial,
        total_partial,
        BLOCK=ZERO_BLOCK,
        num_warps=ZERO_WARPS,
        num_stages=NUM_STAGES,
    )

    _hist_atomic_partition_kernel[(triton.cdiv(N, TILE),)](
        input,
        partial,
        N,
        BINS=num_bins,
        PARTITIONS=PARTITIONS,
        TILE=TILE,
        num_warps=HIST_WARPS,
        num_stages=NUM_STAGES,
    )

    _reduce_partials_kernel[(triton.cdiv(num_bins, BIN_TILE),)](
        partial,
        output,
        BINS=num_bins,
        PARTITIONS=PARTITIONS,
        BIN_TILE=BIN_TILE,
        num_warps=REDUCE_WARPS,
        num_stages=NUM_STAGES,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "MODE": "atomic_partition_reduce",
            "TILE": TILE,
            "PARTITIONS": PARTITIONS,
            "ZERO_BLOCK": ZERO_BLOCK,
            "BIN_TILE": BIN_TILE,
            "zero_num_warps": ZERO_WARPS,
            "hist_num_warps": HIST_WARPS,
            "reduce_num_warps": REDUCE_WARPS,
            "num_stages": NUM_STAGES,
            "atomic_scope": "gpu",
            "atomic_sem": "relaxed",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
