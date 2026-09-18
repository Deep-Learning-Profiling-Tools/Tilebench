```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _zero_i32_kernel(ptr, total: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    tl.store(ptr + offs, tl.zeros((BLOCK,), dtype=tl.int32), mask=mask)


@triton.jit
def _histogram_local_kernel(
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
    part = pid % PARTITIONS

    tl.atomic_add(
        partial_ptr + part * BINS + bins,
        hist,
        sem="relaxed",
        mask=hist != 0,
    )


@triton.jit
def _histogram_atomic_partition_kernel(
    input_ptr,
    partial_ptr,
    N,
    num_bins,
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
    ones = tl.full((BLOCK,), 1, dtype=tl.int32)
    part = pid % PARTITIONS

    tl.atomic_add(
        partial_ptr + part * num_bins + vals,
        ones,
        sem="relaxed",
        mask=mask,
    )


@triton.jit
def _reduce_partials_kernel(
    partial_ptr,
    output_ptr,
    num_bins,
    PARTITIONS: tl.constexpr,
    REDUCE_BINS: tl.constexpr,
):
    pid = tl.program_id(0)
    bins = pid * REDUCE_BINS + tl.arange(0, REDUCE_BINS)
    parts = tl.arange(0, PARTITIONS)

    vals = tl.load(
        partial_ptr + parts[:, None] * num_bins + bins[None, :],
        mask=bins[None, :] < num_bins,
        other=0,
    )
    sums = tl.sum(vals, axis=0)
    tl.store(output_ptr + bins, sums, mask=bins < num_bins)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    PARTITIONS = 256
    ZERO_BLOCK = 1024
    REDUCE_BINS = 16

    ZERO_WARPS = 4
    HIST_WARPS = 8
    ATOMIC_WARPS = 4
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

    if num_bins <= 1024:
        MODE = 0
        INPUT_BLOCK = 8192
        _histogram_local_kernel[(triton.cdiv(N, INPUT_BLOCK),)](
            input,
            partial,
            N,
            BINS=num_bins,
            PARTITIONS=PARTITIONS,
            BLOCK=INPUT_BLOCK,
            num_warps=HIST_WARPS,
            num_stages=NUM_STAGES,
        )
        USED_WARPS = HIST_WARPS
    else:
        MODE = 1
        INPUT_BLOCK = 1024
        _histogram_atomic_partition_kernel[(triton.cdiv(N, INPUT_BLOCK),)](
            input,
            partial,
            N,
            num_bins,
            PARTITIONS=PARTITIONS,
            BLOCK=INPUT_BLOCK,
            num_warps=ATOMIC_WARPS,
            num_stages=NUM_STAGES,
        )
        USED_WARPS = ATOMIC_WARPS

    _reduce_partials_kernel[(triton.cdiv(num_bins, REDUCE_BINS),)](
        partial,
        output,
        num_bins,
        PARTITIONS=PARTITIONS,
        REDUCE_BINS=REDUCE_BINS,
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
            "REDUCE_BINS": REDUCE_BINS,
            "used_num_warps": USED_WARPS,
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


@ct.kernel
def _zero_partial_kernel(partial, ZERO_TILE: ConstInt):
    row = ct.bid(0)
    col = ct.bid(1)
    z = ct.zeros((1, ZERO_TILE), dtype=np.int32)
    ct.store(partial, index=(row, col), tile=z, allow_tma=False)


@ct.kernel
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
    )

    offs = pid * TILE + ct.arange(TILE, dtype=np.int32)
    one = ct.ones((TILE,), dtype=np.int32)
    zero = ct.zeros((TILE,), dtype=np.int32)
    inc = ct.where(offs < N, one, zero)

    part = pid % PARTITIONS
    ct.atomic_add(
        partial,
        (part, vals),
        inc,
        memory_order=ct.MemoryOrder.RELAXED,
        memory_scope=ct.MemoryScope.DEVICE,
    )


@ct.kernel
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
    )
    sums = ct.sum(vals, axis=0)
    ct.store(output, index=(bin_block,), tile=sums, allow_tma=False)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    TILE = 1024
    PARTITIONS = 256
    ZERO_TILE = 256
    BIN_TILE = 16

    ZERO_OCCUPANCY = 8
    HIST_OCCUPANCY = 8
    REDUCE_OCCUPANCY = 8

    partial = torch.empty((PARTITIONS, num_bins), device=input.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    zero_kernel = _zero_partial_kernel.with_hints(occupancy=ZERO_OCCUPANCY)
    ct.launch(
        stream,
        (PARTITIONS, ct.cdiv(num_bins, ZERO_TILE), 1),
        zero_kernel,
        (partial, ZERO_TILE),
    )

    hist_kernel = _hist_atomic_partition_kernel.with_hints(occupancy=HIST_OCCUPANCY)
    ct.launch(
        stream,
        (ct.cdiv(N, TILE), 1, 1),
        hist_kernel,
        (input, partial, N, PARTITIONS, TILE),
    )

    reduce_kernel = _reduce_partials_kernel.with_hints(occupancy=REDUCE_OCCUPANCY)
    ct.launch(
        stream,
        (ct.cdiv(num_bins, BIN_TILE), 1, 1),
        reduce_kernel,
        (partial, output, PARTITIONS, BIN_TILE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "PARTITIONS": PARTITIONS,
            "ZERO_TILE": ZERO_TILE,
            "BIN_TILE": BIN_TILE,
            "zero_occupancy": ZERO_OCCUPANCY,
            "hist_occupancy": HIST_OCCUPANCY,
            "reduce_occupancy": REDUCE_OCCUPANCY,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
