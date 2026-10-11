import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


def _copy_zip_tile(A, B, OUT, start, rows, width):
    a_tile = nl.ndarray((rows, width), dtype=A.dtype, buffer=nl.sbuf)
    b_tile = nl.ndarray((rows, width), dtype=B.dtype, buffer=nl.sbuf)
    zipped = nl.ndarray((rows, 2 * width), dtype=A.dtype, buffer=nl.sbuf)

    nisa.dma_copy(
        dst=a_tile,
        src=A.ap(pattern=[[width, rows], [1, width]], offset=start),
    )
    nisa.dma_copy(
        dst=b_tile,
        src=B.ap(pattern=[[width, rows], [1, width]], offset=start),
    )

    nisa.tensor_copy(dst=zipped[:, 0:2 * width:2], src=a_tile)
    nisa.tensor_copy(dst=zipped[:, 1:2 * width:2], src=b_tile)

    nisa.dma_copy(
        dst=OUT.ap(
            pattern=[[2 * width, rows], [1, 2 * width]],
            offset=2 * start,
        ),
        src=zipped,
    )


@nki.jit
def _interleave_kernel(A, B, N, partitions, width):
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
        _copy_zip_tile(
            A,
            B,
            OUT,
            start + tile * tile_elements,
            partitions,
            width,
        )

    tail_start = start + full_tiles * tile_elements
    remaining = count - full_tiles * tile_elements
    tail_rows = remaining // width

    if tail_rows > 0:
        _copy_zip_tile(A, B, OUT, tail_start, tail_rows, width)

    tail_width = remaining - tail_rows * width
    if tail_width > 0:
        _copy_zip_tile(
            A,
            B,
            OUT,
            tail_start + tail_rows * width,
            1,
            tail_width,
        )

    return OUT


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    partitions = 128
    width = 4000
    n = int(N)

    # FP16 input rows are 64-byte aligned. This fixed width also avoids
    # a separate single-partition tail for the declared input domain.
    _LAST_CONFIG = {
        "lnc": 2,
        "partition_size": partitions,
        "free_size": width,
        "tile_elements": partitions * width,
        "mapping": "sbuf_zip_contiguous_hbm_store",
        "n": n,
    }

    return _interleave_kernel[2](A, B, n, partitions, width)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
