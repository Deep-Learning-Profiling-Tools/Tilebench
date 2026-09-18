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
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    vals = tl.load(
        input_ptr + offs,
        mask=mask,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    part = pid & (PARTITIONS - 1)
    tl.atomic_add(
        partial_ptr + part * BINS + vals,
        tl.full((BLOCK,), 1, dtype=tl.int32),
        sem="relaxed",
        scope="gpu",
        mask=mask,
    )


@triton.jit
def _reduce_partials_bintile_kernel(
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

    PARTITIONS = 128
    INPUT_BLOCK = 1024
    ZERO_BLOCK = 1024
    BIN_TILE = 16

    ZERO_WARPS = 4
    HIST_WARPS = 8
    REDUCE_WARPS = 8
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

    _hist_atomic_partition_kernel[(triton.cdiv(N, INPUT_BLOCK),)](
        input,
        partial,
        N,
        BINS=num_bins,
        PARTITIONS=PARTITIONS,
        BLOCK=INPUT_BLOCK,
        num_warps=HIST_WARPS,
        num_stages=NUM_STAGES,
    )

    _reduce_partials_bintile_kernel[(triton.cdiv(num_bins, BIN_TILE),)](
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
            "MODE": "atomic_partition",
            "PARTITIONS": PARTITIONS,
            "INPUT_BLOCK": INPUT_BLOCK,
            "ZERO_BLOCK": ZERO_BLOCK,
            "BIN_TILE": BIN_TILE,
            "hist_num_warps": HIST_WARPS,
            "zero_num_warps": ZERO_WARPS,
            "reduce_num_warps": REDUCE_WARPS,
            "num_stages": NUM_STAGES,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
