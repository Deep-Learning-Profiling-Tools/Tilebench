import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _copy_tile(A, B, OUT, start, rows, width):
    a_tile = nl.ndarray((rows, width), dtype=A.dtype, buffer=nl.sbuf)
    b_tile = nl.ndarray((rows, width), dtype=B.dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=a_tile,
        src=A.ap(pattern=[[width, rows], [1, width]], offset=start),
    )
    nisa.dma_copy(
        dst=b_tile,
        src=B.ap(pattern=[[width, rows], [1, width]], offset=start),
    )

    nisa.dma_copy(
        dst=OUT.ap(
            pattern=[[2 * width, rows], [2, width]],
            offset=2 * start,
        ),
        src=a_tile,
    )
    nisa.dma_copy(
        dst=OUT.ap(
            pattern=[[2 * width, rows], [2, width]],
            offset=2 * start + 1,
        ),
        src=b_tile,
    )


@nki.jit
def _interleave_kernel(A, B, N, partitions, width):
    # The native dispatcher allocates a fresh output for every invocation.
    OUT = nl.ndarray((2 * N,), dtype=A.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    per_program = N // programs
    start = pid * per_program
    count = per_program
    if pid == programs - 1:
        count = N - start

    tile_elements = partitions * width
    full_tiles = count // tile_elements

    for tile in range(full_tiles):
        _copy_tile(
            A, B, OUT,
            start + tile * tile_elements,
            partitions, width,
        )

    tail_start = start + full_tiles * tile_elements
    remaining = count - full_tiles * tile_elements
    tail_rows = remaining // width

    if tail_rows > 0:
        _copy_tile(A, B, OUT, tail_start, tail_rows, width)

    tail_width = remaining - tail_rows * width
    if tail_width > 0:
        _copy_tile(
            A, B, OUT,
            tail_start + tail_rows * width,
            1, tail_width,
        )

    return OUT


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    partitions = 128
    width = 4096
    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": partitions,
        "free_size": width,
        "tile_elements": partitions * width,
        "mapping": "contiguous_core_split_stride2_dma",
        "n": int(N),
    }
    return _interleave_kernel[2](A, B, N, partitions, width)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
