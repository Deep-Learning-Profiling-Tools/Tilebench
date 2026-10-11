import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Partition count of an on-chip tile (axis 0).
P = 128
# Pipelining constants (the kernel hard-codes the matching slot names b0..b7).
PREFETCH = 5   # tiles loaded ahead of the tile being computed
RING = 8       # SBUF ring slots per program
# Tile sizing: max fp16 elements per partition per tile (16 KiB per row ->
# 2 MiB per tile), tile widths rounded to a multiple of ALIGN elements.
MAX_FREE = 8192
ALIGN = 512
NPROG = 2      # default LNC=2 launch: two programs (one per physical core)

_LAST_CONFIG = {}


def _plan(n):
    # Deterministic function of the shape. Each program owns a contiguous
    # (128, cols) block; split cols into nt nearly equal tiles of <= MAX_FREE
    # columns, with at least 2 tiles so that load/compute/store overlap.
    cols = n // (NPROG * P)
    if cols <= 0:
        return 0, 1, 0
    nt = (cols + MAX_FREE - 1) // MAX_FREE
    if nt < 2 and cols >= 2 * ALIGN:
        nt = 2
    f = (cols + nt - 1) // nt
    f = ((f + ALIGN - 1) // ALIGN) * ALIGN
    if f > cols:
        f = cols
    nt = (cols + f - 1) // f
    return cols, f, nt


def _load_tile(buf, xv, pid, i, F, cols):
    c0 = i * F
    sz = min(F, cols - c0)
    nisa.dma_copy(dst=buf[:, 0:sz], src=xv[pid, :, c0:c0 + sz])


def _relu_store_tile(buf, ov, pid, i, F, cols):
    c0 = i * F
    sz = min(F, cols - c0)
    # VectorE: max(x, 0), in place in SBUF (fp16 in / fp16 out, exact).
    nisa.tensor_scalar(dst=buf[:, 0:sz], data=buf[:, 0:sz], op0=nl.maximum,
                       operand0=0.0, engine=nisa.engine.vector)
    nisa.dma_copy(dst=ov[pid, :, c0:c0 + sz], src=buf[:, 0:sz])


def _process_block(xv, ov, pid, cols, F, dtype):
    # xv / ov: HBM views of shape (nprog, P, cols); this program handles xv[pid].
    nt = (cols + F - 1) // F
    # Fixed ring of 8 SBUF slots; tile i lives in slot (i mod 8).
    b0 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    b1 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    b2 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    b3 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    b4 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    b5 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    b6 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    b7 = nl.ndarray((P, F), dtype=dtype, buffer=nl.sbuf)
    # Prologue: issue the first PREFETCH (=5) loads.
    if nt > 0:
        _load_tile(b0, xv, pid, 0, F, cols)
    if nt > 1:
        _load_tile(b1, xv, pid, 1, F, cols)
    if nt > 2:
        _load_tile(b2, xv, pid, 2, F, cols)
    if nt > 3:
        _load_tile(b3, xv, pid, 3, F, cols)
    if nt > 4:
        _load_tile(b4, xv, pid, 4, F, cols)
    # Steady state: frame invariant b_j <-> tile i + j.
    for i in range(nt):
        if i + 5 < nt:
            # Slot b5 last held tile i - 3, whose store was issued 3 iterations ago.
            _load_tile(b5, xv, pid, i + 5, F, cols)
        _relu_store_tile(b0, ov, pid, i, F, cols)
        tmp = b0
        b0 = b1
        b1 = b2
        b2 = b3
        b3 = b4
        b4 = b5
        b5 = b6
        b6 = b7
        b7 = tmp


@nki.jit
def _relu_kernel(x, free_tile):
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    cols = n // (nprog * P)
    main = cols * nprog * P
    if cols > 0:
        F = min(free_tile, cols)
        if main == n:
            _process_block(x.reshape((nprog, P, cols)),
                           out.reshape((nprog, P, cols)),
                           pid, cols, F, x.dtype)
        else:
            _process_block(x[0:main].reshape((nprog, P, cols)),
                           out[0:main].reshape((nprog, P, cols)),
                           pid, cols, F, x.dtype)
    rem = n - main
    if rem > 0:
        # Tail (< nprog*128 elements); never reached for the configured cases.
        tt = nl.ndarray((1, rem), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=tt, src=x[main:n])
        nisa.tensor_scalar(dst=tt, data=tt, op0=nl.maximum, operand0=0.0,
                           engine=nisa.engine.vector)
        nisa.dma_copy(dst=out[main:n], src=tt)
    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG
    n = x.numel()
    cols, free_tile, nt = _plan(n)
    out = _relu_kernel(x, free_tile)
    _LAST_CONFIG = {
        "programs": NPROG,
        "partitions": P,
        "cols_per_program": cols,
        "tile_free_elems": free_tile,
        "tile_bytes": P * free_tile * 2,
        "tiles_per_program": nt,
        "prefetch_tiles": PREFETCH,
        "sbuf_ring_slots": RING,
        "layout": "(programs, 128, n/(programs*128)) contiguous rows per partition",
        "compute": "VectorE max(x,0) in-place SBUF, software-pipelined DMA ring",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
