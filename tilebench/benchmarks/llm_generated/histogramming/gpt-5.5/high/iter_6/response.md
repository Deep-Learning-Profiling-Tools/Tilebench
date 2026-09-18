```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _zero_partial_kernel(partial, ZERO_TILE: ConstInt):
    row = ct.bid(0)
    col = ct.bid(1)
    z = ct.zeros((1, ZERO_TILE), dtype=np.int32)
    ct.store(partial, index=(row, col), tile=z, allow_tma=False)


@ct.kernel(occupancy=2)
def _hist_persistent_partition_kernel(
    input,
    partial,
    N: ConstInt,
    NUM_TILES: ConstInt,
    PARTITIONS: ConstInt,
    TILE: ConstInt,
    LOOPS: ConstInt,
):
    pid = ct.bid(0)
    lanes = ct.arange(TILE, dtype=np.int32)
    one = ct.ones((TILE,), dtype=np.int32)
    zero = ct.zeros((TILE,), dtype=np.int32)

    for j in range(0, LOOPS):
        tile_id = pid + j * PARTITIONS
        if tile_id < NUM_TILES:
            vals = ct.load(
                input,
                index=(tile_id,),
                shape=(TILE,),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
                latency=1,
            )
            offs = tile_id * TILE + lanes
            inc = ct.where(offs < N, one, zero)
            ct.atomic_add(
                partial,
                (pid, vals),
                inc,
                check_bounds=False,
                memory_order=ct.MemoryOrder.RELAXED,
                memory_scope=ct.MemoryScope.BLOCK,
            )


@ct.kernel(occupancy=8)
def _reduce_groups_kernel(
    partial,
    temp,
    GROUP: ConstInt,
    BIN_TILE: ConstInt,
):
    bin_block = ct.bid(0)
    group = ct.bid(1)

    vals = ct.load(
        partial,
        index=(group, bin_block),
        shape=(GROUP, BIN_TILE),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
        latency=1,
    )
    sums = ct.sum(vals, axis=0)
    ct.store(
        temp,
        index=(group, bin_block),
        tile=ct.reshape(sums, (1, BIN_TILE)),
        allow_tma=False,
    )


@ct.kernel(occupancy=8)
def _reduce_final_kernel(
    temp,
    output,
    GROUPS: ConstInt,
    BIN_TILE: ConstInt,
):
    bin_block = ct.bid(0)

    vals = ct.load(
        temp,
        index=(0, bin_block),
        shape=(GROUPS, BIN_TILE),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
        latency=1,
    )
    sums = ct.sum(vals, axis=0)
    ct.store(output, index=(bin_block,), tile=sums, allow_tma=False)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    TILE = 4096
    PARTITIONS = 1024
    GROUP = 64
    GROUPS = PARTITIONS // GROUP
    ZERO_TILE = 2048
    BIN_TILE = 16
    HIST_OCCUPANCY = 2
    AUX_OCCUPANCY = 8

    num_tiles = ct.cdiv(N, TILE)
    loops = ct.cdiv(num_tiles, PARTITIONS)

    partial = torch.empty((PARTITIONS, num_bins), device=input.device, dtype=torch.int32)
    temp = torch.empty((GROUPS, num_bins), device=input.device, dtype=torch.int32)

    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (PARTITIONS, ct.cdiv(num_bins, ZERO_TILE), 1),
        _zero_partial_kernel,
        (partial, ZERO_TILE),
    )

    ct.launch(
        stream,
        (PARTITIONS, 1, 1),
        _hist_persistent_partition_kernel,
        (input, partial, N, num_tiles, PARTITIONS, TILE, loops),
    )

    ct.launch(
        stream,
        (ct.cdiv(num_bins, BIN_TILE), GROUPS, 1),
        _reduce_groups_kernel,
        (partial, temp, GROUP, BIN_TILE),
    )

    ct.launch(
        stream,
        (ct.cdiv(num_bins, BIN_TILE), 1, 1),
        _reduce_final_kernel,
        (temp, output, GROUPS, BIN_TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "MODE": "persistent_partition_block_atomic_twostage_reduce",
            "TILE": TILE,
            "PARTITIONS": PARTITIONS,
            "GROUP": GROUP,
            "GROUPS": GROUPS,
            "ZERO_TILE": ZERO_TILE,
            "BIN_TILE": BIN_TILE,
            "hist_occupancy": HIST_OCCUPANCY,
            "aux_occupancy": AUX_OCCUPANCY,
            "atomic_check_bounds": False,
            "atomic_scope": "block",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
