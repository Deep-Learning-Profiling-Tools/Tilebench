```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _zero_i32_kernel(ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    z = tl.zeros((BLOCK,), dtype=tl.int32)
    tl.store(ptr + offs, z, mask=offs < total)


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
def _reduce_partials_chunk_atomic_kernel(
    partial_ptr,
    output_ptr,
    BINS: tl.constexpr,
    PARTITIONS: tl.constexpr,
    PART_BLOCK: tl.constexpr,
    BIN_TILE: tl.constexpr,
):
    bin_pid = tl.program_id(0)
    part_pid = tl.program_id(1)

    bins = bin_pid * BIN_TILE + tl.arange(0, BIN_TILE)
    parts = part_pid * PART_BLOCK + tl.arange(0, PART_BLOCK)

    vals = tl.load(
        partial_ptr + parts[:, None] * BINS + bins[None, :],
        mask=(parts[:, None] < PARTITIONS) & (bins[None, :] < BINS),
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    sums = tl.sum(vals, axis=0)

    tl.atomic_add(
        output_ptr + bins,
        sums,
        sem="relaxed",
        scope="gpu",
        mask=bins < BINS,
    )


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    TILE = 512
    PARTITIONS = 128
    PART_BLOCK = 32
    BIN_TILE = 16
    ZERO_BLOCK = 1024

    ZERO_WARPS = 4
    HIST_WARPS = 8
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

    _zero_i32_kernel[(triton.cdiv(num_bins, ZERO_BLOCK),)](
        output,
        num_bins,
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

    _reduce_partials_chunk_atomic_kernel[
        (triton.cdiv(num_bins, BIN_TILE), triton.cdiv(PARTITIONS, PART_BLOCK))
    ](
        partial,
        output,
        BINS=num_bins,
        PARTITIONS=PARTITIONS,
        PART_BLOCK=PART_BLOCK,
        BIN_TILE=BIN_TILE,
        num_warps=REDUCE_WARPS,
        num_stages=NUM_STAGES,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "MODE": "atomic_partition_reduce_chunked",
            "TILE": TILE,
            "PARTITIONS": PARTITIONS,
            "PART_BLOCK": PART_BLOCK,
            "BIN_TILE": BIN_TILE,
            "ZERO_BLOCK": ZERO_BLOCK,
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
def _hist_atomic_partition_nomask_kernel(
    input,
    partial,
    PARTITIONS: ConstInt,
    TILE: ConstInt,
):
    pid = ct.bid(0)
    vals = ct.load(
        input,
        index=(pid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.UNDETERMINED,
        allow_tma=False,
        latency=1,
    )

    one = ct.ones((TILE,), dtype=np.int32)
    part = pid & (PARTITIONS - 1)

    ct.atomic_add(
        partial,
        (part, vals),
        one,
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

    TILE = 2048
    PARTITIONS = 32
    ZERO_TILE = 4096
    BIN_TILE = 64
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
        (N // TILE, 1, 1),
        _hist_atomic_partition_nomask_kernel,
        (input, partial, PARTITIONS, TILE),
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
            "MODE": "atomic_partition_reduce_nomask",
            "TILE": TILE,
            "PARTITIONS": PARTITIONS,
            "ZERO_TILE": ZERO_TILE,
            "BIN_TILE": BIN_TILE,
            "occupancy": OCCUPANCY,
            "atomic_check_bounds": False,
            "atomic_scope": "device",
            "atomic_order": "relaxed",
            "assume_N_multiple_of_TILE": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
