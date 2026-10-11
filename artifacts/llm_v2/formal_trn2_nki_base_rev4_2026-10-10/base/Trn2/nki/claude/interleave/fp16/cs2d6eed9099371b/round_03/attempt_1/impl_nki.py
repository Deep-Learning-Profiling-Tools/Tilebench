import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Static configuration (literals). The block plan is a fixed deterministic
# function of the per-core element count.
# ---------------------------------------------------------------------------
ROWS = 128         # partitions per block (always full width; remainder -> tail)
F_MAX = 4096       # max elements per partition per block (per input)
F_MIN = 512        # min elements per partition per block
F_ALIGN = 256      # block width multiple (256 fp16 = 512 B aligned descriptors)
NB_MIN = 12        # aim for at least this many blocks per core
N_PROGRAMS = 2     # LNC=2: two physical cores per logical core

_LAST_CONFIG = {}


def _plan(C, rows, f_max, f_min, f_align, nb_min):
    L = C // rows                 # elements per partition handled by blocks
    tail = C - L * rows           # < rows leftover elements (single partition)
    F = (L + nb_min - 1) // nb_min
    F = ((F + f_align - 1) // f_align) * f_align
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
    nb = nfull
    if flast > 0:
        nb = nb + 1
    D = 2
    if F <= 2048:
        D = 3
    if F <= 1024:
        D = 4
    return L, tail, F, nfull, flast, nb, D


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
def interleave_kernel(A, B, rows, f_max, f_min, f_align, nb_min):
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

    L, tail, F, nfull, flast, nb, D = _plan(cnt, rows, f_max, f_min, f_align, nb_min)
    blk = rows * F

    # small single-partition tail (independent; issued first so it overlaps)
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
        a2 = a0
        b2 = b0
        a3 = a0
        b3 = b0
        a4 = a0
        b4 = b0
        # prologue: blocks 1 .. D-1
        for i in range(1, D):
            if i < nb:
                wi = _bw(i, nfull, F, flast)
                pa = _load(A, start + i * blk, rows, wi, dt)
                pb = _load(B, start + i * blk, rows, wi, dt)
                if i == 1:
                    a1 = pa
                    b1 = pb
                if i == 2:
                    a2 = pa
                    b2 = pb
                if i == 3:
                    a3 = pa
                    b3 = pb
        for j in range(nb):
            if j + D < nb:
                wn = _bw(j + D, nfull, F, flast)
                na = _load(A, start + (j + D) * blk, rows, wn, dt)
                nbt = _load(B, start + (j + D) * blk, rows, wn, dt)
                if D == 2:
                    a2 = na
                    b2 = nbt
                if D == 3:
                    a3 = na
                    b3 = nbt
                if D == 4:
                    a4 = na
                    b4 = nbt
            wj = _bw(j, nfull, F, flast)
            use_vec = (j % 2) == 0
            _zip_store(out, a0, b0, start + j * blk, rows, wj, dt, use_vec)
            a0 = a1
            b0 = b1
            a1 = a2
            b1 = b2
            a2 = a3
            b2 = b3
            a3 = a4
            b3 = b4
    return out


def _config_for(n):
    per = n // N_PROGRAMS
    cores = []
    for pid in range(N_PROGRAMS):
        start = pid * per
        cnt = per if pid < N_PROGRAMS - 1 else n - start
        L, tail, F, nfull, flast, nb, D = _plan(cnt, ROWS, F_MAX, F_MIN, F_ALIGN, NB_MIN)
        cores.append({
            "start": start,
            "count": cnt,
            "rows": ROWS,
            "elems_per_partition": L,
            "F": F,
            "full_blocks": nfull,
            "last_block_F": flast,
            "blocks": nb,
            "prefetch": D,
            "tail": tail,
        })
    return {
        "kernel": "interleave_kernel",
        "n": n,
        "rows": ROWS,
        "f_max": F_MAX,
        "f_min": F_MIN,
        "f_align": F_ALIGN,
        "nb_min": NB_MIN,
        "programs": N_PROGRAMS,
        "zip_engines": "alternate vector/scalar per block",
        "cores": cores,
    }


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(A.shape[0])
    out = interleave_kernel(A, B, ROWS, F_MAX, F_MIN, F_ALIGN, NB_MIN)
    _LAST_CONFIG = _config_for(n)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
