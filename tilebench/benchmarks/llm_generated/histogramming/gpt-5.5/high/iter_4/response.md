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
def _histogram_direct_kernel(
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

    hist = tl.histogram(vals, BINS, mask=mask)
    bins = tl.arange(0, BINS)
    part = pid & (PARTITIONS - 1)

    tl.atomic_add(
        partial_ptr + part * BINS + bins,
        hist,
        sem="relaxed",
        scope="gpu",
        mask=hist != 0,
    )


@triton.jit
def _histogram_split256_kernel(
    input_ptr,
    partial_ptr,
    N,
    BINS: tl.constexpr,
    PARTITIONS: tl.constexpr,
    BLOCK: tl.constexpr,
    GROUPS: tl.constexpr,
    SUB_BINS: tl.constexpr,
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

    low = vals & 255
    high = vals >> 8
    bins = tl.arange(0, SUB_BINS)
    part = pid & (PARTITIONS - 1)
    base = partial_ptr + part * BINS

    for g in tl.range(0, GROUPS, 1):
        gmask = mask & (high == g)
        hist = tl.histogram(low, SUB_BINS, mask=gmask)
        tl.atomic_add(
            base + g * SUB_BINS + bins,
            hist,
            sem="relaxed",
            scope="gpu",
            mask=hist != 0,
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

    PARTITIONS = 256
    ZERO_BLOCK = 1024
    INPUT_BLOCK = 4096
    BIN_TILE = 8
    SUB_BINS = 256

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

    if num_bins <= SUB_BINS:
        MODE = 0
        GROUPS = 1
        _histogram_direct_kernel[(triton.cdiv(N, INPUT_BLOCK),)](
            input,
            partial,
            N,
            BINS=num_bins,
            PARTITIONS=PARTITIONS,
            BLOCK=INPUT_BLOCK,
            num_warps=HIST_WARPS,
            num_stages=NUM_STAGES,
        )
    else:
        MODE = 1
        GROUPS = num_bins // SUB_BINS
        _histogram_split256_kernel[(triton.cdiv(N, INPUT_BLOCK),)](
            input,
            partial,
            N,
            BINS=num_bins,
            PARTITIONS=PARTITIONS,
            BLOCK=INPUT_BLOCK,
            GROUPS=GROUPS,
            SUB_BINS=SUB_BINS,
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
            "MODE": MODE,
            "PARTITIONS": PARTITIONS,
            "INPUT_BLOCK": INPUT_BLOCK,
            "ZERO_BLOCK": ZERO_BLOCK,
            "BIN_TILE": BIN_TILE,
            "SUB_BINS": SUB_BINS,
            "GROUPS": GROUPS,
            "hist_num_warps": HIST_WARPS,
            "zero_num_warps": ZERO_WARPS,
            "reduce_num_warps": REDUCE_WARPS,
            "num_stages": NUM_STAGES,
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


@ct.kernel(occupancy=8)
def _hist_atomic_partition_kernel(
    input,
    partial,
    N: ConstInt,
    PARTITIONS: ConstInt,
    TILE: ConstInt,
):
    pid = ct.bid(0)
    vals = ct.load(
        input,
        index=(pid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
        latency=1,
    )

    offs = pid * TILE + ct.arange(TILE, dtype=np.int32)
    one = ct.ones((TILE,), dtype=np.int32)
    zero = ct.zeros((TILE,), dtype=np.int32)
    inc = ct.where(offs < N, one, zero)

    part = pid & (PARTITIONS - 1)
    ct.atomic_add(
        partial,
        (part, vals),
        inc,
        check_bounds=False,
        memory_order=ct.MemoryOrder.RELAXED,
        memory_scope=ct.MemoryScope.DEVICE,
    )


@ct.kernel(occupancy=8)
def _reduce_partials_kernel(
    partial,
    output,
    PARTITIONS: ConstInt,
    BIN_TILE: ConstInt,
):
    bin_block = ct.bid(0)
    vals = ct.load(
        partial,
        index=(0, bin_block),
        shape=(PARTITIONS, BIN_TILE),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
        latency=1,
    )
    sums = ct.sum(vals, axis=0)
    ct.store(output, index=(bin_block,), tile=sums, allow_tma=False)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    TILE = 1024
    PARTITIONS = 256
    ZERO_TILE = 2048
    BIN_TILE = 8
    OCCUPANCY = 8

    partial = torch.empty((PARTITIONS, num_bins), device=input.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (PARTITIONS, ct.cdiv(num_bins, ZERO_TILE), 1),
        _zero_partial_kernel,
        (partial, ZERO_TILE),
    )

    ct.launch(
        stream,
        (ct.cdiv(N, TILE), 1, 1),
        _hist_atomic_partition_kernel,
        (input, partial, N, PARTITIONS, TILE),
    )

    ct.launch(
        stream,
        (ct.cdiv(num_bins, BIN_TILE), 1, 1),
        _reduce_partials_kernel,
        (partial, output, PARTITIONS, BIN_TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "PARTITIONS": PARTITIONS,
            "ZERO_TILE": ZERO_TILE,
            "BIN_TILE": BIN_TILE,
            "occupancy": OCCUPANCY,
            "atomic_check_bounds": False,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
