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
def _hist_persistent_partition_kernel(
    input_ptr,
    partial_ptr,
    N,
    BINS: tl.constexpr,
    PARTITIONS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(N, BLOCK)
    lanes = tl.arange(0, BLOCK)
    one = tl.full((BLOCK,), 1, dtype=tl.int32)

    for tile_id in tl.range(pid, n_tiles, PARTITIONS, num_stages=1):
        offs = tile_id * BLOCK + lanes
        mask = offs < N
        vals = tl.load(
            input_ptr + offs,
            mask=mask,
            other=0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        tl.atomic_add(
            partial_ptr + pid * BINS + vals,
            one,
            sem="relaxed",
            scope="cta",
            mask=mask,
        )


@triton.jit
def _reduce_groups_kernel(
    partial_ptr,
    temp_ptr,
    BINS: tl.constexpr,
    GROUP: tl.constexpr,
    BIN_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    gid = tl.program_id(1)

    bins = bid * BIN_TILE + tl.arange(0, BIN_TILE)
    parts = gid * GROUP + tl.arange(0, GROUP)

    vals = tl.load(
        partial_ptr + parts[:, None] * BINS + bins[None, :],
        mask=bins[None, :] < BINS,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    sums = tl.sum(vals, axis=0)
    tl.store(temp_ptr + gid * BINS + bins, sums, mask=bins < BINS)


@triton.jit
def _reduce_final_kernel(
    temp_ptr,
    output_ptr,
    BINS: tl.constexpr,
    GROUPS: tl.constexpr,
    BIN_TILE: tl.constexpr,
):
    bid = tl.program_id(0)

    bins = bid * BIN_TILE + tl.arange(0, BIN_TILE)
    groups = tl.arange(0, GROUPS)

    vals = tl.load(
        temp_ptr + groups[:, None] * BINS + bins[None, :],
        mask=bins[None, :] < BINS,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    sums = tl.sum(vals, axis=0)
    tl.store(output_ptr + bins, sums, mask=bins < BINS)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    PARTITIONS = 1024
    GROUP = 64
    GROUPS = PARTITIONS // GROUP
    INPUT_BLOCK = 1024
    ZERO_BLOCK = 1024
    BIN_TILE = 16

    ZERO_WARPS = 4
    HIST_WARPS = 8
    REDUCE1_WARPS = 8
    REDUCE2_WARPS = 4
    NUM_STAGES = 2

    partial = torch.empty((PARTITIONS, num_bins), device=input.device, dtype=torch.int32)
    temp = torch.empty((GROUPS, num_bins), device=input.device, dtype=torch.int32)

    total_partial = PARTITIONS * num_bins

    _zero_i32_kernel[(triton.cdiv(total_partial, ZERO_BLOCK),)](
        partial,
        total_partial,
        BLOCK=ZERO_BLOCK,
        num_warps=ZERO_WARPS,
        num_stages=NUM_STAGES,
    )

    _hist_persistent_partition_kernel[(PARTITIONS,)](
        input,
        partial,
        N,
        BINS=num_bins,
        PARTITIONS=PARTITIONS,
        BLOCK=INPUT_BLOCK,
        num_warps=HIST_WARPS,
        num_stages=NUM_STAGES,
    )

    _reduce_groups_kernel[(triton.cdiv(num_bins, BIN_TILE), GROUPS)](
        partial,
        temp,
        BINS=num_bins,
        GROUP=GROUP,
        BIN_TILE=BIN_TILE,
        num_warps=REDUCE1_WARPS,
        num_stages=NUM_STAGES,
    )

    _reduce_final_kernel[(triton.cdiv(num_bins, BIN_TILE),)](
        temp,
        output,
        BINS=num_bins,
        GROUPS=GROUPS,
        BIN_TILE=BIN_TILE,
        num_warps=REDUCE2_WARPS,
        num_stages=NUM_STAGES,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "MODE": "persistent_partition_cta_atomic_twostage_reduce",
            "PARTITIONS": PARTITIONS,
            "GROUP": GROUP,
            "GROUPS": GROUPS,
            "INPUT_BLOCK": INPUT_BLOCK,
            "ZERO_BLOCK": ZERO_BLOCK,
            "BIN_TILE": BIN_TILE,
            "hist_num_warps": HIST_WARPS,
            "zero_num_warps": ZERO_WARPS,
            "reduce1_num_warps": REDUCE1_WARPS,
            "reduce2_num_warps": REDUCE2_WARPS,
            "num_stages": NUM_STAGES,
            "atomic_scope": "cta",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
