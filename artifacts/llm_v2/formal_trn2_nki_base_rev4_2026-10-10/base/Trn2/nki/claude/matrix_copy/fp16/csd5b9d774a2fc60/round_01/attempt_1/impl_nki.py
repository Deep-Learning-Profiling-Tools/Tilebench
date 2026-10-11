import torch

import nki
import nki.language as nl
import nki.isa as nisa


# Rows of A moved by one DMA instruction (each row is contiguous; the chunk
# A[r0:r0+ROWS_PER_DMA, :] is a contiguous byte range of the matrix).
ROWS_PER_DMA = 128

_LAST_CONFIG = {}


@nki.jit
def _matrix_copy_kernel(a, rows_per_dma):
    R = a.shape[0]
    C = a.shape[1]
    # Stage 1: allocate the output (uninitialised).
    out = nl.ndarray((R, C), dtype=a.dtype, buffer=nl.shared_hbm)

    # Split the rows between the programs (2 physical cores under LNC=2).
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rows_per_prog = (R + n_prog - 1) // n_prog
    start = pid * rows_per_prog
    end = min(R, start + rows_per_prog)
    n_chunks = (end - start + rows_per_dma - 1) // rows_per_dma

    # Stage 2: single streaming traversal, each element moved exactly once,
    # directly HBM -> HBM (no intermediate global scratch).
    for c in range(n_chunks):
        r0 = start + c * rows_per_dma
        r1 = min(end, r0 + rows_per_dma)
        nisa.dma_copy(dst=out[r0:r1, :], src=a[r0:r1, :])
    return out


def run(A: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    rows = int(A.shape[0])
    cols = int(A.shape[1])
    rows_per_dma = ROWS_PER_DMA
    B = _matrix_copy_kernel(A, rows_per_dma)
    n_prog_assumed = 2
    rows_per_prog = (rows + n_prog_assumed - 1) // n_prog_assumed
    _LAST_CONFIG = {
        "path": "hbm_to_hbm_dma",
        "rows_per_dma": rows_per_dma,
        "programs": n_prog_assumed,
        "rows_per_program": rows_per_prog,
        "dmas_per_program": (rows_per_prog + rows_per_dma - 1) // rows_per_dma,
        "shape": [rows, cols],
    }
    return B


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
