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
def _hist_atomic_partition_nomask_kernel(
    input_ptr,
    partial_ptr,
    BINS: tl.constexpr,
    PARTITIONS: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    lanes = tl.arange(0, TILE)
    offs = pid * TILE + lanes
    vals = tl.load(input_ptr + offs)

    part = pid & (PARTITIONS - 1)
    one = tl.full((TILE,), 1, dtype=tl.int32)

    tl.atomic_add(
        partial_ptr + part * BINS + vals,
        one,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def _hist_atomic_partition_masked_kernel(
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
    vals = tl.load(input_ptr + offs, mask=mask, other=0)

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
def _reduce_partials_loop_kernel(
    partial_ptr,
    output_ptr,
    BINS: tl.constexpr,
    PARTITIONS: tl.constexpr,
    BIN_TILE: tl.constexpr,
):
    bin_pid = tl.program_id(0)
    bins = bin_pid * BIN_TILE + tl.arange(0, BIN_TILE)
    valid = bins < BINS

    acc = tl.zeros((BIN_TILE,), dtype=tl.int32)
    for p in tl.range(0, PARTITIONS, 1):
        vals = tl.load(
            partial_ptr + p * BINS + bins,
            mask=valid,
            other=0,
        )
        acc += vals

    tl.store(output_ptr + bins, acc, mask=valid)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    TILE = 1024
    PARTITIONS = 128
    ZERO_BLOCK = 1024
    BIN_TILE = 16

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

    assume_multiple = (N % TILE) == 0
    if assume_multiple:
        _hist_atomic_partition_nomask_kernel[(N // TILE,)](
            input,
            partial,
            BINS=num_bins,
            PARTITIONS=PARTITIONS,
            TILE=TILE,
            num_warps=HIST_WARPS,
            num_stages=NUM_STAGES,
        )
    else:
        _hist_atomic_partition_masked_kernel[(triton.cdiv(N, TILE),)](
            input,
            partial,
            N,
            BINS=num_bins,
            PARTITIONS=PARTITIONS,
            TILE=TILE,
            num_warps=HIST_WARPS,
            num_stages=NUM_STAGES,
        )

    _reduce_partials_loop_kernel[(triton.cdiv(num_bins, BIN_TILE),)](
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
            "MODE": "atomic_partition_reduce_loop_nomask" if assume_multiple else "atomic_partition_reduce_loop_masked",
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
            "assume_N_multiple_of_TILE": assume_multiple,
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
def _hist_atomic_partition_masked_kernel(
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
    PARTITIONS = 128
    ZERO_TILE = 4096
    BIN_TILE = 16
    OCCUPANCY = 8

    partial = torch.empty((PARTITIONS, num_bins), device=input.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        (PARTITIONS, ct.cdiv(num_bins, ZERO_TILE), 1),
        _zero_partial_kernel,
        (partial, ZERO_TILE),
    )

    assume_multiple = (N % TILE) == 0
    if assume_multiple:
        ct.launch(
            stream,
            (N // TILE, 1, 1),
            _hist_atomic_partition_nomask_kernel,
            (input, partial, PARTITIONS, TILE),
        )
    else:
        ct.launch(
            stream,
            (ct.cdiv(N, TILE), 1, 1),
            _hist_atomic_partition_masked_kernel,
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
            "MODE": "atomic_partition_reduce_nomask" if assume_multiple else "atomic_partition_reduce_masked",
            "TILE": TILE,
            "PARTITIONS": PARTITIONS,
            "ZERO_TILE": ZERO_TILE,
            "BIN_TILE": BIN_TILE,
            "occupancy": OCCUPANCY,
            "atomic_check_bounds": False,
            "atomic_scope": "device",
            "atomic_order": "relaxed",
            "assume_N_multiple_of_TILE": assume_multiple,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
