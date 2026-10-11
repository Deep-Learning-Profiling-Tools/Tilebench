import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _copy_zip_group(
    A, B, OUT, start, total_rows, first_row, partitions, width, group_size
):
    a_tiles = nl.ndarray(
        (partitions, group_size, width), dtype=A.dtype, buffer=nl.sbuf
    )
    b_tiles = nl.ndarray(
        (partitions, group_size, width), dtype=B.dtype, buffer=nl.sbuf
    )
    zipped = nl.ndarray(
        (partitions, group_size, 2 * width), dtype=A.dtype, buffer=nl.sbuf
    )

    # Enqueue the next tile's loads before the current tile's store.
    # The DMA and vector instruction streams can then overlap.
    for tile in range(group_size):
        row = first_row + tile * partitions
        rows = min(partitions, total_rows - row)
        if rows > 0:
            offset = start + row * width
            nisa.dma_copy(
                dst=a_tiles[0:rows, tile, :],
                src=A.ap(pattern=[[width, rows], [1, width]], offset=offset),
            )
            nisa.dma_copy(
                dst=b_tiles[0:rows, tile, :],
                src=B.ap(pattern=[[width, rows], [1, width]], offset=offset),
            )

    for tile in range(group_size):
        row = first_row + tile * partitions
        rows = min(partitions, total_rows - row)
        if rows > 0:
            nisa.tensor_copy(
                dst=zipped[0:rows, tile, 0:2 * width:2],
                src=a_tiles[0:rows, tile, :],
                engine=nisa.engine.vector,
            )
            nisa.tensor_copy(
                dst=zipped[0:rows, tile, 1:2 * width:2],
                src=b_tiles[0:rows, tile, :],
                engine=nisa.engine.vector,
            )

    for tile in range(group_size):
        row = first_row + tile * partitions
        rows = min(partitions, total_rows - row)
        if rows > 0:
            offset = start + row * width
            nisa.dma_copy(
                dst=OUT.ap(
                    pattern=[[2 * width, rows], [1, 2 * width]],
                    offset=2 * offset,
                ),
                src=zipped[0:rows, tile, :],
            )


def _copy_zip_tail(A, B, OUT, start, width):
    a_tile = nl.ndarray((1, width), dtype=A.dtype, buffer=nl.sbuf)
    b_tile = nl.ndarray((1, width), dtype=B.dtype, buffer=nl.sbuf)
    zipped = nl.ndarray((1, 2 * width), dtype=A.dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=a_tile,
        src=A.ap(pattern=[[width, 1], [1, width]], offset=start),
    )
    nisa.dma_copy(
        dst=b_tile,
        src=B.ap(pattern=[[width, 1], [1, width]], offset=start),
    )
    nisa.tensor_copy(
        dst=zipped[:, 0:2 * width:2],
        src=a_tile,
        engine=nisa.engine.vector,
    )
    nisa.tensor_copy(
        dst=zipped[:, 1:2 * width:2],
        src=b_tile,
        engine=nisa.engine.vector,
    )
    nisa.dma_copy(
        dst=OUT.ap(
            pattern=[[2 * width, 1], [1, 2 * width]],
            offset=2 * start,
        ),
        src=zipped,
    )


@nki.jit
def _interleave_kernel(A, B, N, partitions, width, group_size):
    OUT = nl.ndarray((2 * N,), dtype=A.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    per_program = N // programs
    start = pid * per_program
    count = per_program
    if pid == programs - 1:
        count = N - start

    total_rows = count // width
    rows_per_group = partitions * group_size
    groups = (total_rows + rows_per_group - 1) // rows_per_group

    for group in range(groups):
        _copy_zip_group(
            A, B, OUT,
            start,
            total_rows,
            group * rows_per_group,
            partitions,
            width,
            group_size,
        )

    tail_width = count - total_rows * width
    if tail_width > 0:
        _copy_zip_tail(
            A, B, OUT,
            start + total_rows * width,
            tail_width,
        )

    return OUT


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    partitions = 128
    max_width = 8192
    group_size = 2

    # Balance the tiles while retaining aligned, contiguous DMA rows.
    count = (int(N) + 1) // 2
    target_tiles = (
        count + partitions * max_width - 1
    ) // (partitions * max_width)
    denominator = partitions * target_tiles
    required_width = (count + denominator - 1) // denominator
    width = ((required_width + 255) // 256) * 256

    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": partitions,
        "free_size": width,
        "max_free_size": max_width,
        "prefetch_tiles": group_size,
        "tile_elements": partitions * width,
        "mapping": "prefetched_vector_zip_contiguous_hbm_store",
        "n": int(N),
    }

    return _interleave_kernel[2](
        A, B, int(N), partitions, width, group_size
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
