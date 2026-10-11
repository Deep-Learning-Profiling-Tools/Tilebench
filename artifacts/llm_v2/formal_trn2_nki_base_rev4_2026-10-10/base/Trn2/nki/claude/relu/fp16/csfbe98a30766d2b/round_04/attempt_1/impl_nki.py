import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Partition count of an on-chip tile (axis 0).
P = 128
NPROG = 2        # default LNC=2 launch: two programs (one per physical core)
# Tile schedule (columns per partition; fp16 -> 2 bytes per column):
MAXF = 8192      # largest tile: 128 x 8192 fp16 = 2 MiB
ALIGN = 512      # middle tiles are multiples of 512 columns
HEAD0 = 2048     # head ramp: 2048, 4096, 8192, ...
TAIL0 = 512      # tail ramp (reversed at the end): ..., 4096, 2048, 1024, 512
# Pipelining constants (the kernel hard-codes the matching slot names b0..b7).
PREFETCH = 5     # tiles loaded ahead of the tile being computed
RING = 8         # SBUF ring slots per program

_LAST_CONFIG = {}


# ---------------------------------------------------------------------------
# Trace-time schedule helpers (pure integer logic; usable from Python and from
# inside the kernel).  Deterministic function of cols = n / (programs * 128).
# ---------------------------------------------------------------------------
def _pow2(k):
    r = 1
    while k > 0:
        r = r * 2
        k = k - 1
    return r


def _ramp_count(cols):
    # Number of head (= number of tail) ramp tiles.
    rem = cols
    i = 0
    go = True
    while go:
        wl = min(MAXF, HEAD0 * _pow2(i))
        wr = min(MAXF, TAIL0 * _pow2(i))
        if wl + wr > rem:
            go = False
        else:
            rem = rem - wl - wr
            i = i + 1
            if wl >= MAXF:
                if wr >= MAXF:
                    go = False
            if i >= 8:
                go = False
    return i


def _ramp_sum(cnt, base):
    s = 0
    i = 0
    while i < cnt:
        s = s + min(MAXF, base * _pow2(i))
        i = i + 1
    return s


def _mid_rem(cols):
    nr = _ramp_count(cols)
    return cols - _ramp_sum(nr, HEAD0) - _ramp_sum(nr, TAIL0)


def _mid_count(cols):
    rem = _mid_rem(cols)
    res = 0
    if rem > 0:
        res = (rem + MAXF - 1) // MAXF
    return res


def _num_tiles(cols):
    return 2 * _ramp_count(cols) + _mid_count(cols)


def _tile_width(cols, t):
    nr = _ramp_count(cols)
    m = _mid_count(cols)
    res = 0
    if t < nr:
        res = min(MAXF, HEAD0 * _pow2(t))
    else:
        j = t - nr
        if j < m:
            rem = _mid_rem(cols)
            units = rem // ALIGN
            leftover = rem - units * ALIGN
            mm = max(m, 1)
            per = units // mm
            extra = units - per * mm
            w = per * ALIGN
            if j < extra:
                w = w + ALIGN
            if j == m - 1:
                w = w + leftover
            res = w
        else:
            k = nr - 1 - (j - m)
            res = min(MAXF, TAIL0 * _pow2(k))
    return res


def _tile_offset(cols, t):
    s = 0
    j = 0
    while j < t:
        s = s + _tile_width(cols, j)
        j = j + 1
    return s


# ---------------------------------------------------------------------------
# Kernel pieces
# ---------------------------------------------------------------------------
def _load_tile(buf, xv, pid, cols, t):
    c0 = _tile_offset(cols, t)
    w = _tile_width(cols, t)
    nisa.dma_copy(dst=buf[:, 0:w], src=xv[pid, :, c0:c0 + w])


def _relu_store_tile(buf, ov, pid, cols, t):
    c0 = _tile_offset(cols, t)
    w = _tile_width(cols, t)
    # VectorE: max(x, 0), in place in SBUF (fp16 in / fp16 out, exact).
    nisa.tensor_scalar(dst=buf[:, 0:w], data=buf[:, 0:w], op0=nl.maximum,
                       operand0=0.0, engine=nisa.engine.vector)
    nisa.dma_copy(dst=ov[pid, :, c0:c0 + w], src=buf[:, 0:w])


def _process_block(xv, ov, pid, cols, dtype):
    # xv / ov: HBM views of shape (nprog, P, cols); this program handles xv[pid].
    nt = _num_tiles(cols)
    F = min(MAXF, cols)
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
        _load_tile(b0, xv, pid, cols, 0)
    if nt > 1:
        _load_tile(b1, xv, pid, cols, 1)
    if nt > 2:
        _load_tile(b2, xv, pid, cols, 2)
    if nt > 3:
        _load_tile(b3, xv, pid, cols, 3)
    if nt > 4:
        _load_tile(b4, xv, pid, cols, 4)
    # Steady state: frame invariant b_j <-> tile i + j.
    for i in range(nt):
        if i + 5 < nt:
            # Slot b5 last held tile i - 3, whose store was issued 3 iterations ago.
            _load_tile(b5, xv, pid, cols, i + 5)
        _relu_store_tile(b0, ov, pid, cols, i)
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
def _relu_kernel(x):
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    cols = n // (nprog * P)
    main = cols * nprog * P
    if cols > 0:
        if main == n:
            _process_block(x.reshape((nprog, P, cols)),
                           out.reshape((nprog, P, cols)),
                           pid, cols, x.dtype)
        else:
            _process_block(x[0:main].reshape((nprog, P, cols)),
                           out[0:main].reshape((nprog, P, cols)),
                           pid, cols, x.dtype)
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
    out = _relu_kernel(x)
    cols = n // (NPROG * P)
    widths = []
    if cols > 0:
        nt = _num_tiles(cols)
        for t in range(nt):
            widths.append(int(_tile_width(cols, t)))
    _LAST_CONFIG = {
        "programs": NPROG,
        "partitions": P,
        "cols_per_program": int(cols),
        "tiles_per_program": len(widths),
        "tile_widths": widths,
        "max_tile_free_elems": MAXF,
        "head_ramp_start": HEAD0,
        "tail_ramp_end": TAIL0,
        "mid_align": ALIGN,
        "prefetch_tiles": PREFETCH,
        "sbuf_ring_slots": RING,
        "layout": "(programs, 128, n/(programs*128)) contiguous rows per partition",
        "compute": "VectorE max(x,0) in-place SBUF, graded-tile software-pipelined DMA ring",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
