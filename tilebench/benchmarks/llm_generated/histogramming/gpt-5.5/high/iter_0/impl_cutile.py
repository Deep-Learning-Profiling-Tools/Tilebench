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
