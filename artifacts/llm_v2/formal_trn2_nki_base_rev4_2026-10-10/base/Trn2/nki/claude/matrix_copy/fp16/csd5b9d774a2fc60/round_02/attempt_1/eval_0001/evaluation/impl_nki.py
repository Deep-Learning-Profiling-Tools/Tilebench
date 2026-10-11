import torch

import nki
import nki.language as nl
import nki.isa as nisa


# ---------------------------------------------------------------------------
# Tunables (fixed literals; the plan below is a deterministic function of the
# shape and element size only).
# ---------------------------------------------------------------------------
TARGET_DMA_BYTES = 4 * 1024 * 1024   # upper bound for bytes moved by one dma_copy
MIN_ROW_BYTES = 2048                 # minimum contiguous bytes per DMA row
MIN_ROWS_PER_DMA = 32                # never use fewer rows than this per DMA
MAX_ROWS_PER_DMA = 128               # rows (outer dim) per DMA instruction
ASSUMED_PROGRAMS = 2                 # LNC=2: two physical cores per launch

_LAST_CONFIG = {}


@nki.jit
def _matrix_copy_kernel(a, view_rows, view_cols, rows_per_dma):
    R = a.shape[0]
    C = a.shape[1]
    # Stage 1: allocate the output (uninitialised).
    out = nl.ndarray((R, C), dtype=a.dtype, buffer=nl.shared_hbm)

    # Both A and B are contiguous row-major with identical layouts, so the
    # copy is expressed over a flat 2-D view (metadata only, no copy).
    a2 = a.reshape((view_rows, view_cols))
    o2 = out.reshape((view_rows, view_cols))

    # Split the view rows between the programs (2 physical cores under LNC=2).
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rows_per_prog = (view_rows + n_prog - 1) // n_prog
    start = pid * rows_per_prog
    end = min(view_rows, start + rows_per_prog)
    n_chunks = (end - start + rows_per_dma - 1) // rows_per_dma

    # Stage 2: single streaming traversal, each element moved exactly once,
    # directly HBM -> HBM (no intermediate global scratch).
    for c in range(n_chunks):
        r0 = start + c * rows_per_dma
        r1 = min(end, r0 + rows_per_dma)
        nisa.dma_copy(dst=o2[r0:r1, :], src=a2[r0:r1, :])
    return out


def _plan(rows, cols, esz):
    """Deterministic copy plan from shape + element size."""
    total = rows * cols
    P = MAX_ROWS_PER_DMA
    if total % ASSUMED_PROGRAMS == 0 and (total // ASSUMED_PROGRAMS) % P == 0:
        L = total // ASSUMED_PROGRAMS            # elements per program
        n_dma = 1
        # Double the number of DMAs (exact split) while each one is too large.
        while (L * esz) // n_dma > TARGET_DMA_BYTES and L % (P * n_dma * 2) == 0:
            n_dma *= 2
        F = L // (P * n_dma)
        # For small matrices, use fewer, longer rows (exact: halving P doubles F).
        while F * esz < MIN_ROW_BYTES and P > MIN_ROWS_PER_DMA and P % 2 == 0:
            P //= 2
            F *= 2
        view_rows = total // F
        return {
            "path": "flat_hbm_to_hbm_dma",
            "view_rows": view_rows,
            "view_cols": F,
            "rows_per_dma": P,
            "dmas_per_program": n_dma,
            "bytes_per_dma": P * F * esz,
            "row_bytes": F * esz,
        }
    # Generic fallback (not needed for the configured domain): original rows,
    # 128-row chunks with remainder handling inside the kernel.
    rows_per_prog = (rows + ASSUMED_PROGRAMS - 1) // ASSUMED_PROGRAMS
    return {
        "path": "row_hbm_to_hbm_dma",
        "view_rows": rows,
        "view_cols": cols,
        "rows_per_dma": MAX_ROWS_PER_DMA,
        "dmas_per_program": (rows_per_prog + MAX_ROWS_PER_DMA - 1) // MAX_ROWS_PER_DMA,
        "bytes_per_dma": MAX_ROWS_PER_DMA * cols * esz,
        "row_bytes": cols * esz,
    }


def run(A: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    rows = int(A.shape[0])
    cols = int(A.shape[1])
    esz = int(A.element_size())
    plan = _plan(rows, cols, esz)
    B = _matrix_copy_kernel(
        A, int(plan["view_rows"]), int(plan["view_cols"]), int(plan["rows_per_dma"])
    )
    cfg = dict(plan)
    cfg["programs"] = ASSUMED_PROGRAMS
    cfg["shape"] = [rows, cols]
    _LAST_CONFIG = cfg
    return B


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
