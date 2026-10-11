import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Static configuration (literals). The block plan is a fixed deterministic
# function of the per-core element count.
# ---------------------------------------------------------------------------
F_MAX = 4096       # max elements per partition per block (per input)
F_MIN = 512        # min elements per partition per block (DMA efficiency)
NB_MIN = 12        # aim for at least this many blocks per core (pipelining)
RAMP_MIN_F = 1024  # split the final full block in two halves when F >= this
PREFETCH = 3       # blocks loaded ahead of the block being zipped/stored
N_PROGRAMS = 2     # LNC=2: two physical cores per logical core

_LAST_CONFIG = {}


def _plan(C, f_max, f_min, nb_min):
    # Largest partition count in [64, 128] dividing C (no tail needed);
    # otherwise 128 partitions plus a single-partition tail.
    rows = 0
    for i in range(65):
        r = 128 - i
        if rows == 0:
            if C % r == 0:
                rows = r
    tail = 0
    if rows == 0:
        rows = 128
        tail = C % 128
    L = (C - tail) // rows          # elements per partition (whole core range)
    F = (L + nb_min - 1) // nb_min
    if F < f_min:
        F = f_min
    if F > f_max:
        F = f_max
    if F > L:
        F = L
    nfull = 0
    flast = 0
    if F > 0:
        nfull = L // F
        flast = L - nfull * F
    return rows, L, F, nfull, flast, tail


def _sched(F, nfull, flast, ramp_min):
    ramp = 0
    if F >= ramp_min:
        if nfull >= 2:
            ramp = 1
    h1 = 0
    if ramp == 1:
        h1 = (F // 2) // 2 * 2
    nb = nfull
    if flast > 0:
        nb = nb + 1
    if ramp == 1:
        nb = nb + 1
    return ramp, h1, nb


def _blk(j, F, nfull, flast, ramp, h1):
    # (column offset within the core range in units of `rows`, width)
    # ramp order: [F]*(nfull-1), flast (if any), h1, F-h1
    c0 = 0
    w = 0
    if ramp == 1:
        nmain = nfull - 1
        extra = 0
        if flast > 0:
            extra = 1
        if j < nmain:
            c0 = j * F
            w = F
        else:
            if j < nmain + extra:
                c0 = nmain * F
                w = flast
            else:
                k = j - nmain - extra
                base = nmain * F + flast
                if k == 0:
                    c0 = base
                    w = h1
                else:
                    c0 = base + h1
                    w = F - h1
    else:
        if j < nfull:
            c0 = j * F
            w = F
        else:
            c0 = nfull * F
            w = flast
    return c0, w


def _load(X, off, rows, w, dt):
    # contiguous block X[off : off + rows*w] -> SBUF [rows, w]
    t = nl.ndarray((rows, w), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=X.ap(pattern=[[w, rows], [1, w]], offset=off))
    return t


def _zip_store(O, a_t, b_t, off, rows, w, dt):
    # o_t[p, 2k] = a_t[p, k], o_t[p, 2k+1] = b_t[p, k]; one contiguous store.
    # Columns are split between VectorE (first half) and ScalarE (second half).
    o_t = nl.ndarray((rows, 2 * w), dtype=dt, buffer=nl.sbuf)
    h = (w // 2) // 2 * 2
    if h >= 2:
        nisa.tensor_copy(dst=o_t[:, 0:2 * h:2], src=a_t[:, 0:h], engine=nisa.engine.vector)
        nisa.tensor_copy(dst=o_t[:, 1:2 * h:2], src=b_t[:, 0:h], engine=nisa.engine.vector)
        nisa.tensor_copy(dst=o_t[:, 2 * h:2 * w:2], src=a_t[:, h:w], engine=nisa.engine.scalar)
        nisa.tensor_copy(dst=o_t[:, 2 * h + 1:2 * w:2], src=b_t[:, h:w], engine=nisa.engine.scalar)
    else:
        nisa.tensor_copy(dst=o_t[:, 0:2 * w:2], src=a_t, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=o_t[:, 1:2 * w:2], src=b_t, engine=nisa.engine.vector)
    nisa.dma_copy(dst=O.ap(pattern=[[2 * w, rows], [1, 2 * w]], offset=2 * off), src=o_t)


@nki.jit
def interleave_kernel(A, B, f_max, f_min, nb_min, ramp_min):
    N = A.shape[0]
    dt = A.dtype
    out = nl.ndarray((2 * N,), dtype=dt, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    per = N // n_prog
    start = pid * per
    cnt = per
    if pid == n_prog - 1:
        cnt = N - start

    rows, L, F, nfull, flast, tail = _plan(cnt, f_max, f_min, nb_min)
    ramp, h1, nb = _sched(F, nfull, flast, ramp_min)

    # generic single-partition tail (never taken when rows divides cnt)
    if tail > 0:
        toff = start + rows * L
        ta = _load(A, toff, 1, tail, dt)
        tb = _load(B, toff, 1, tail, dt)
        _zip_store(out, ta, tb, toff, 1, tail, dt)

    if nb > 0:
        c0, w0 = _blk(0, F, nfull, flast, ramp, h1)
        a0 = _load(A, start + rows * c0, rows, w0, dt)
        b0 = _load(B, start + rows * c0, rows, w0, dt)
        a1 = a0
        b1 = b0
        if nb > 1:
            c1, w1 = _blk(1, F, nfull, flast, ramp, h1)
            a1 = _load(A, start + rows * c1, rows, w1, dt)
            b1 = _load(B, start + rows * c1, rows, w1, dt)
        a2 = a1
        b2 = b1
        if nb > 2:
            c2, w2 = _blk(2, F, nfull, flast, ramp, h1)
            a2 = _load(A, start + rows * c2, rows, w2, dt)
            b2 = _load(B, start + rows * c2, rows, w2, dt)
        for j in range(nb):
            a3 = a2
            b3 = b2
            if j + 3 < nb:
                c3, w3 = _blk(j + 3, F, nfull, flast, ramp, h1)
                a3 = _load(A, start + rows * c3, rows, w3, dt)
                b3 = _load(B, start + rows * c3, rows, w3, dt)
            cj, wj = _blk(j, F, nfull, flast, ramp, h1)
            _zip_store(out, a0, b0, start + rows * cj, rows, wj, dt)
            a0 = a1
            b0 = b1
            a1 = a2
            b1 = b2
            a2 = a3
            b2 = b3
    return out


def _config_for(n):
    per = n // N_PROGRAMS
    cores = []
    for pid in range(N_PROGRAMS):
        start = pid * per
        cnt = per if pid < N_PROGRAMS - 1 else n - start
        rows, L, F, nfull, flast, tail = _plan(cnt, F_MAX, F_MIN, NB_MIN)
        ramp, h1, nb = _sched(F, nfull, flast, RAMP_MIN_F)
        widths = []
        for j in range(nb):
            _, w = _blk(j, F, nfull, flast, ramp, h1)
            widths.append(w)
        cores.append({
            "start": start,
            "count": cnt,
            "rows": rows,
            "elems_per_partition": L,
            "F": F,
            "full_blocks": nfull,
            "last_block_F": flast,
            "ramp_down": ramp,
            "blocks": nb,
            "block_widths": widths,
            "tail": tail,
        })
    return {
        "kernel": "interleave_kernel",
        "n": n,
        "f_max": F_MAX,
        "f_min": F_MIN,
        "nb_min": NB_MIN,
        "ramp_min_f": RAMP_MIN_F,
        "prefetch": PREFETCH,
        "programs": N_PROGRAMS,
        "zip_engines": "per block: vector first half of columns, scalar second half",
        "cores": cores,
    }


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(A.shape[0])
    out = interleave_kernel(A, B, F_MAX, F_MIN, NB_MIN, RAMP_MIN_F)
    _LAST_CONFIG = _config_for(n)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
