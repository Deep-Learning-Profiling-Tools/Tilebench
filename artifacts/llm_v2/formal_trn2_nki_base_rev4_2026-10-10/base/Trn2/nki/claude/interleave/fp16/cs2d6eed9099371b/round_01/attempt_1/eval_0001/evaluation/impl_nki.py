import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Static configuration (literals; the plan below is a fixed deterministic
# function of the per-core element count).
# ---------------------------------------------------------------------------
F_MAX = 4096      # max elements per partition per block (per input)
F_MIN = 512       # min elements per partition per block (DMA efficiency)
NB_MIN = 12       # aim for at least this many blocks per core (pipelining)
PREFETCH = 2      # blocks loaded ahead of the block being zipped/stored
N_PROGRAMS = 2    # LNC=2: two physical cores per logical core

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


def _bw(j, nfull, F, flast):
    w = flast
    if j < nfull:
        w = F
    return w


def _load(X, off, rows, w, dt):
    # contiguous block X[off : off + rows*w] -> SBUF [rows, w]
    t = nl.ndarray((rows, w), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=X.ap(pattern=[[w, rows], [1, w]], offset=off))
    return t


def _zip_store(O, a_t, b_t, off, rows, w, dt, use_vector):
    # o_t[p, 2k] = a_t[p, k], o_t[p, 2k+1] = b_t[p, k]; then one contiguous store
    o_t = nl.ndarray((rows, 2 * w), dtype=dt, buffer=nl.sbuf)
    if use_vector:
        nisa.tensor_copy(dst=o_t[:, 0:2 * w:2], src=a_t, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=o_t[:, 1:2 * w:2], src=b_t, engine=nisa.engine.vector)
    else:
        nisa.tensor_copy(dst=o_t[:, 0:2 * w:2], src=a_t, engine=nisa.engine.scalar)
        nisa.tensor_copy(dst=o_t[:, 1:2 * w:2], src=b_t, engine=nisa.engine.scalar)
    nisa.dma_copy(dst=O.ap(pattern=[[2 * w, rows], [1, 2 * w]], offset=2 * off), src=o_t)


@nki.jit
def interleave_kernel(A, B, f_max, f_min, nb_min):
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
    nb = nfull
    if flast > 0:
        nb = nfull + 1
    blk = rows * F

    # generic single-partition tail (never taken when rows divides cnt)
    if tail > 0:
        toff = start + rows * L
        ta = _load(A, toff, 1, tail, dt)
        tb = _load(B, toff, 1, tail, dt)
        _zip_store(out, ta, tb, toff, 1, tail, dt, True)

    if nb > 0:
        w0 = _bw(0, nfull, F, flast)
        a0 = _load(A, start, rows, w0, dt)
        b0 = _load(B, start, rows, w0, dt)
        a1 = a0
        b1 = b0
        if nb > 1:
            w1 = _bw(1, nfull, F, flast)
            a1 = _load(A, start + blk, rows, w1, dt)
            b1 = _load(B, start + blk, rows, w1, dt)
        for j in range(nb):
            a2 = a1
            b2 = b1
            if j + 2 < nb:
                w2 = _bw(j + 2, nfull, F, flast)
                a2 = _load(A, start + (j + 2) * blk, rows, w2, dt)
                b2 = _load(B, start + (j + 2) * blk, rows, w2, dt)
            wj = _bw(j, nfull, F, flast)
            use_vec = (j % 2) == 0
            _zip_store(out, a0, b0, start + j * blk, rows, wj, dt, use_vec)
            a0 = a1
            b0 = b1
            a1 = a2
            b1 = b2
    return out


def _config_for(n):
    per = n // N_PROGRAMS
    cores = []
    for pid in range(N_PROGRAMS):
        start = pid * per
        cnt = per if pid < N_PROGRAMS - 1 else n - start
        rows, L, F, nfull, flast, tail = _plan(cnt, F_MAX, F_MIN, NB_MIN)
        nb = nfull + (1 if flast > 0 else 0)
        cores.append({
            "start": start,
            "count": cnt,
            "rows": rows,
            "elems_per_partition": L,
            "F": F,
            "full_blocks": nfull,
            "last_block_F": flast,
            "blocks": nb,
            "tail": tail,
        })
    return {
        "kernel": "interleave_kernel",
        "n": n,
        "f_max": F_MAX,
        "f_min": F_MIN,
        "nb_min": NB_MIN,
        "prefetch": PREFETCH,
        "programs": N_PROGRAMS,
        "zip_engines": "alternate vector/scalar per block",
        "cores": cores,
    }


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(A.shape[0])
    out = interleave_kernel(A, B, F_MAX, F_MIN, NB_MIN)
    _LAST_CONFIG = _config_for(n)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
