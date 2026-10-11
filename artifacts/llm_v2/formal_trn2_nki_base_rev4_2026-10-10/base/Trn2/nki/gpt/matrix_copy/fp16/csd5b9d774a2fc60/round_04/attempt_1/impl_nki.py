import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}
_SINGLE_CORE_MAX_BYTES = 1048576
_PARTITIONS = 32
_MAX_TILE_FREE = 32768
_FREE_ALIGNMENT = 256


@nki.jit
def _matrix_copy_kernel(A, partitions, tile_free, tile_count):
    B = nl.ndarray(A.shape, dtype=A.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)
    elements_per_core = A.shape[0] * A.shape[1] // programs
    core_offset = pid * elements_per_core
    tile_elements = partitions * tile_free

    if tile_count == 1:
        free = elements_per_core // partitions
        nisa.dma_copy(
            dst=B.ap(
                pattern=[[free, partitions], [1, free]],
                offset=core_offset,
            ),
            src=A.ap(
                pattern=[[free, partitions], [1, free]],
                offset=core_offset,
            ),
        )
    else:
        buffers = nl.ndarray(
            (partitions, 2, tile_free),
            dtype=A.dtype,
            buffer=nl.sbuf,
        )

        nisa.dma_copy(
            dst=buffers[:, 0, :],
            src=A.ap(
                pattern=[[tile_free, partitions], [1, tile_free]],
                offset=core_offset,
            ),
        )

        for tile in range(tile_count):
            start = tile * tile_elements
            free = min(
                tile_free,
                (elements_per_core - start) // partitions,
            )

            if tile + 1 < tile_count:
                next_start = (tile + 1) * tile_elements
                next_free = min(
                    tile_free,
                    (elements_per_core - next_start) // partitions,
                )
                nisa.dma_copy(
                    dst=buffers[:, (tile + 1) % 2, :next_free],
                    src=A.ap(
                        pattern=[[next_free, partitions], [1, next_free]],
                        offset=core_offset + next_start,
                    ),
                )

            nisa.dma_copy(
                dst=B.ap(
                    pattern=[[free, partitions], [1, free]],
                    offset=core_offset + start,
                ),
                src=buffers[:, tile % 2, :free],
            )

    return B


def run(A: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    elements = N * N
    lnc = 1 if elements * 2 <= _SINGLE_CORE_MAX_BYTES else 2
    elements_per_core = elements // lnc
    free_size = elements_per_core // _PARTITIONS

    tile_count = (free_size + _MAX_TILE_FREE - 1) // _MAX_TILE_FREE
    alignment_span = tile_count * _FREE_ALIGNMENT
    tile_free = (
        (free_size + alignment_span - 1) // alignment_span
    ) * _FREE_ALIGNMENT

    _LAST_CONFIG = {
        "strategy": (
            "direct_hbm_dma_copy"
            if tile_count == 1
            else "double_buffered_stream_copy"
        ),
        "lnc": lnc,
        "partition_size": _PARTITIONS,
        "free_size": free_size,
        "tile_free_size": tile_free,
        "tiles_per_core": tile_count,
        "buffers": 0 if tile_count == 1 else 2,
        "elements_per_core": elements_per_core,
        "single_core_max_bytes": _SINGLE_CORE_MAX_BYTES,
        "max_tile_free": _MAX_TILE_FREE,
        "free_alignment": _FREE_ALIGNMENT,
    }

    return _matrix_copy_kernel[lnc](
        A, _PARTITIONS, tile_free, tile_count
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
