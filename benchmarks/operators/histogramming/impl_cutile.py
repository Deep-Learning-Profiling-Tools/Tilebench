import math
import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

_LAST_CONFIG = None


@ct.kernel
def histogram_partial_kernel(
    input,
    partial,
    N: ConstInt,
    num_bins: ConstInt,
    num_partials: ConstInt,
    BLOCK_SIZE: ConstInt,
):
    pid = ct.bid(0)

    chunk_idx = pid
    while chunk_idx * BLOCK_SIZE < N:
        vals = ct.load(
            input,
            index=(chunk_idx,),
            shape=(BLOCK_SIZE,),
            padding_mode=ct.PaddingMode.ZERO,
        )

        offs = chunk_idx * BLOCK_SIZE + ct.arange(BLOCK_SIZE, dtype=ct.int32)
        valid = offs < N
        in_range = ct.bitwise_and(vals >= 0, vals < num_bins)
        valid = ct.bitwise_and(valid, in_range)

        row_idx = ct.full((BLOCK_SIZE,), pid, dtype=ct.int32)
        bin_idx = ct.where(valid, vals, 0)
        update = ct.astype(valid, ct.int32)

        ct.atomic_add(partial, (row_idx, bin_idx), update)

        chunk_idx = chunk_idx + num_partials


@ct.kernel
def histogram_reduce_kernel(
    partial,
    histogram,
    num_partials: ConstInt,
    num_bins: ConstInt,
    BLOCK_ROWS: ConstInt,
    BLOCK_BINS: ConstInt,
):
    pid_b = ct.bid(0)

    acc = ct.full((BLOCK_BINS,), 0, dtype=ct.int32)

    num_row_tiles = ct.cdiv(num_partials, BLOCK_ROWS)
    row_tile = 0
    while row_tile < num_row_tiles:
        tile = ct.load(
            partial,
            index=(row_tile, pid_b),
            shape=(BLOCK_ROWS, BLOCK_BINS),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = acc + ct.sum(tile, axis=0)
        row_tile = row_tile + 1

    ct.store(histogram, index=(pid_b,), tile=acc)


def run(
    input,
    N: int,
    num_bins: int,
    BLOCK_SIZE: int = 1024,
    NUM_PARTIAL: int = 256,
    BLOCK_ROWS: int = 64,
    BLOCK_BINS: int = 256,
    block_size: int = None,
    autotune: bool = False,
):
    global _LAST_CONFIG

    if block_size is not None:
        BLOCK_SIZE = int(block_size)

    BLOCK_SIZE = int(BLOCK_SIZE)
    NUM_PARTIAL = int(NUM_PARTIAL)
    BLOCK_ROWS = int(BLOCK_ROWS)
    BLOCK_BINS = int(BLOCK_BINS)

    assert input.is_cuda
    assert input.ndim == 1
    assert input.shape[0] == N
    assert input.dtype == torch.int32
    assert num_bins >= 1

    input = input.contiguous()

    num_partials = min(NUM_PARTIAL, math.ceil(N / BLOCK_SIZE))
    partial = torch.zeros((num_partials, num_bins), device=input.device, dtype=torch.int32)
    histogram = torch.empty((num_bins,), device=input.device, dtype=torch.int32)

    ct.launch(
        torch.cuda.current_stream(),
        (num_partials, 1, 1),
        histogram_partial_kernel,
        (
            input,
            partial,
            N,
            num_bins,
            num_partials,
            BLOCK_SIZE,
        ),
    )

    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(num_bins / BLOCK_BINS), 1, 1),
        histogram_reduce_kernel,
        (
            partial,
            histogram,
            num_partials,
            num_bins,
            BLOCK_ROWS,
            BLOCK_BINS,
        ),
    )

    _LAST_CONFIG = {
        "BLOCK_SIZE": BLOCK_SIZE,
        "NUM_PARTIAL": NUM_PARTIAL,
        "BLOCK_ROWS": BLOCK_ROWS,
        "BLOCK_BINS": BLOCK_BINS,
    }
    return histogram


def get_last_config() -> dict | None:
    return _LAST_CONFIG
