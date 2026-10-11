import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}
_SINGLE_CORE_MAX_BYTES = 1048576


@nki.jit
def _matrix_copy_kernel(A, free_size):
    B = nl.ndarray(A.shape, dtype=A.dtype, buffer=nl.shared_hbm)

    programs = nl.num_programs()
    pid = nl.program_id(0)

    src = A.reshape((programs, 128, free_size))
    dst = B.reshape((programs, 128, free_size))

    nisa.dma_copy(dst=dst[pid, :, :], src=src[pid, :, :])
    return B


def run(A: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG

    elements = N * N
    matrix_bytes = elements * 2
    lnc = 1 if matrix_bytes <= _SINGLE_CORE_MAX_BYTES else 2
    free_size = elements // (128 * lnc)

    _LAST_CONFIG = {
        "strategy": "direct_hbm_dma_copy",
        "lnc": lnc,
        "partition_size": 128,
        "free_size": free_size,
        "tiles_per_core": 1,
        "elements_per_core": elements // lnc,
        "single_core_max_bytes": _SINGLE_CORE_MAX_BYTES,
    }

    return _matrix_copy_kernel[lnc](A, free_size)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
