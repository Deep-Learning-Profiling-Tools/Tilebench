import torch

import nki
import nki.language as nl
import nki.isa as nisa


# ---------------------------------------------------------------------------
# Tunables (fixed literals). The copy plan is a deterministic function of the
# shape and element size only.
# ---------------------------------------------------------------------------
MAX_ROWS_PER_DMA = 128      # outer (partition-like) rows per DMA -> spread over DMA engines
MIN_ROWS_PER_DMA = 32       # never fewer outer rows than this
MIN_ROW_BYTES = 2048        # minimum contiguous bytes per outer row (small matrices)
MAX_SUB_BYTES = 32768       # max contiguous bytes of one innermost run (descriptor-friendly)
ALIGN_BYTES = 64            # preferred byte alignment of innermost runs
ASSUMED_PROGRAMS = 2        # LNC=2: two physical cores per launch

_LAST_CONFIG = {}


@nki.jit
def _matrix_copy_kernel(a, sub, cols, rows_per_dma):
    R = a.shape[0]
    C = a.shape[1]
    # Stage 1: allocate the output (uninitialised).
    out = nl.ndarray((R, C), dtype=a.dtype, buffer=nl.shared_hbm)

    # A and B are contiguous row-major with identical layouts: express the copy
    # over a flat view (metadata only).  view = [view_rows, sub, cols].
    view_rows = (R * C) // (sub * cols)

    # Split the outer rows between the programs (two physical cores).
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rows_per_prog = (view_rows + n_prog - 1) // n_prog
    start = pid * rows_per_prog
    end = min(view_rows, start + rows_per_prog)
    n_chunks = 0
    if end > start:
        n_chunks = (end - start + rows_per_dma - 1) // rows_per_dma

    # Stage 2: single streaming traversal, every element moved exactly once,
    # directly HBM -> HBM (no intermediate global scratch).
    if sub == 1:
        a2 = a.reshape((view_rows, cols))
        o2 = out.reshape((view_rows, cols))
        for c in range(n_chunks):
            r0 = start + c * rows_per_dma
            r1 = min(end, r0 + rows_per_dma)
            nisa.dma_copy(dst=o2[r0:r1, :], src=a2[r0:r1, :])
    else:
        a3 = a.reshape((view_rows, sub, cols))
        o3 = out.reshape((view_rows, sub, cols))
        for c in range(n_chunks):
            r0 = start + c * rows_per_dma
            r1 = min(end, r0 + rows_per_dma)
            nisa.dma_copy(dst=o3[r0:r1, :, :], src=a3[r0:r1, :, :])
    return out


def _pick_sub(row_elems, esz):
    """Smallest split k of one outer row into k equal contiguous runs, each at
    most MAX_SUB_BYTES, preferring 64-byte aligned run lengths."""
    max_sub = max(1, MAX_SUB_BYTES // esz)
    align = max(1, ALIGN_BYTES // esz)
    if row_elems <= max_sub:
        return 1
    k0 = (row_elems + max_sub - 1) // max_sub
    k = k0
    while k <= row_elems:
        if row_elems % k == 0 and (row_elems // k) % align == 0:
            return k
        k += 1
    k = k0
    while k <= row_elems:
        if row_elems % k == 0:
            return k
        k += 1
    return row_elems


def _plan(rows, cols, esz):
    total = rows * cols
    progs = ASSUMED_PROGRAMS
    if total % progs == 0:
        L = total // progs                      # elements per program
        P = MAX_ROWS_PER_DMA
        while P > MIN_ROWS_PER_DMA and (L % P != 0 or (L // P) * esz < MIN_ROW_BYTES):
            P //= 2
        if L % P == 0:
            row_elems = L // P                  # contiguous elements per outer row
            k = _pick_sub(row_elems, esz)
            F = row_elems // k
            return {
                "path": "flat_hbm_to_hbm_single_dma",
                "rows_per_dma": P,
                "sub": k,
                "view_cols": F,
                "view_rows": total // (k * F),
                "dmas_per_program": 1,
                "bytes_per_dma": P * k * F * esz,
                "run_bytes": F * esz,
            }
    # Generic fallback (not reached in the configured domain).
    rows_per_prog = (rows + progs - 1) // progs
    return {
        "path": "row_hbm_to_hbm_dma",
        "rows_per_dma": MAX_ROWS_PER_DMA,
        "sub": 1,
        "view_cols": cols,
        "view_rows": rows,
        "dmas_per_program": (rows_per_prog + MAX_ROWS_PER_DMA - 1) // MAX_ROWS_PER_DMA,
        "bytes_per_dma": MAX_ROWS_PER_DMA * cols * esz,
        "run_bytes": cols * esz,
    }


def run(A: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    rows = int(A.shape[0])
    cols = int(A.shape[1])
    esz = int(A.element_size())
    plan = _plan(rows, cols, esz)
    B = _matrix_copy_kernel(
        A, int(plan["sub"]), int(plan["view_cols"]), int(plan["rows_per_dma"])
    )
    cfg = dict(plan)
    cfg["programs"] = ASSUMED_PROGRAMS
    cfg["shape"] = [rows, cols]
    _LAST_CONFIG = cfg
    return B


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
