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
    )
    sums = ct.sum(vals, axis=0)
    ct.store(output, index=(bin_block,), tile=sums, allow_tma=False)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    TILE = 1024
    PARTITIONS = 1024
    ZERO_TILE = 1024
    BIN_TILE = 1
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
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
