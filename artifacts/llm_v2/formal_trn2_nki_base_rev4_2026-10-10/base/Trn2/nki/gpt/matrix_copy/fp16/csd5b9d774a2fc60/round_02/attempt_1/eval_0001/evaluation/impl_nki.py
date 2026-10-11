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

    src = A.reshape((programs, 128, free_size))
    dst = B.reshape((programs, 128, free_size))

    # Disjoint contiguous halves, copied directly by the DMA primitive.
    nisa.dma_copy(
        dst=dst[pid, :, :],
        src=src[pid, :, :],
    )
    return B


def run(A: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    free_size = (N * N) // 256
    _LAST_CONFIG = {
        "strategy": "direct_hbm_dma_copy",
        "lnc": 2,
        "partition_size": 128,
        "free_size": free_size,
        "tiles_per_core": 1,
        "elements_per_core": 128 * free_size,
    }

    return _matrix_copy_kernel[2](A, free_size)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
