import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _matrix_copy_kernel(A, free_size):
    B = nl.ndarray(A.shape, dtype=A.dtype, buffer=nl.shared_hbm)

    pid = nl.program_id(0)
    programs = nl.num_programs()
    partitions = 128

    # Each core owns one contiguous half of the flattened matrix.
    src = A.reshape((programs, partitions, free_size))
    dst = B.reshape((programs, partitions, free_size))

    tile = nl.ndarray(
        (partitions, free_size),
        dtype=A.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(dst=tile, src=src[pid, :, :])
    nisa.dma_copy(dst=dst[pid, :, :], src=tile)
    return B


def run(A: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    # N is a multiple of 256, so both halves divide exactly into 128
    # partitions. At N=5120 this uses 204800 bytes per partition,
    # below the documented 212984-byte usable SBUF capacity.
    free_size = (N * N) // 256

    _LAST_CONFIG = {
        "strategy": "contiguous_bulk_copy",
        "lnc": 2,
        "partition_size": 128,
        "free_size": free_size,
        "tiles_per_core": 1,
        "elements_per_core": 128 * free_size,
    }

    return _matrix_copy_kernel[2](A, free_size)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
