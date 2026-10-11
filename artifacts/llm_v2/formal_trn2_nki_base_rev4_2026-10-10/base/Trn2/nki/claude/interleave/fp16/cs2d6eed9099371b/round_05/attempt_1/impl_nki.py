import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ---------------------------------------------------------------------------
# Static configuration (literals). The block plan is a fixed deterministic
# function of the per-core element count.
# ---------------------------------------------------------------------------
ROWS = 128         # partitions per block (remainder -> single-partition tail)
F_MAX = 4096       # max elements per partition per main block (per input)
F_MIN = 512        # min elements per partition per main block
F_ALIGN = 256      # main block width multiple
NB_MIN = 12        # aim for at least this many main blocks per core
W_MIN = 512        # min width of head / taper blocks
R_MIN = 256        # min width of a standalone leftover block
N_PROGRAMS = 2     # LNC=2: two physical cores per logical core

_LAST_CONFIG = {}


def _plan(C, rows, f_max, f_min, f_align, nb_min, w_min, r_min):
    L = C // rows                 # elements per partition handled by blocks
    tail = C - L * rows           # < rows leftover elements (single partition)
    F = 0
    if L > 0:
        F = (L + nb_min - 1) // nb_min
        F = ((F + f_align - 1) // f_align) * f_align
        if F < f_min:
            F = f_min
        if F > f_max:
            F = f_max
        if F > L:
            F = L
    h = 0
    x = 0
    nmain = 0
    t1 = 0
    t2 = 0
    if F > 0:
        hh = F // 4
        if hh < w_min:
            hh = w_min
        q1 = F // 2
        if q1 < w_min:
            q1 = w_min
        q2 = F // 4
        if q2 < w_min:
            q2 = w_min
        if L >= hh + q1 + q2 + F:
            rem = L - hh - q1 - q2
            nmain = rem // F
            r = rem - nmain * F
            h = hh
            t1 = q1
            t2 = q2
            if r > 0:
                if h + r <= F:
                    h = h + r
                elif r >= r_min:
                    x = r
                else:
                    t1 = t1 + r
        else:
            nmain = L // F
            t1 = L - nmain * F
    nb = nmain
    if h > 0:
        nb = nb + 1
    if x > 0:
        nb = nb + 1
    if t1 > 0:
        nb = nb + 1
    if t2 > 0:
        nb = nb + 1
    D = 2
    if F <= 2048:
        D = 3
    if F <= 1024:
        D = 4
    return L, tail, F, h, x, nmain, t1, t2, nb, D


def _bw(j, h, x, nmain, F, t1, t2):
    # width of block j in the sequence [h] [x] F*nmain [t1] [t2]
    w = 0
    k = j
    done = 0
    if h > 0:
        if k == 0:
            w = h
            done = 1
        k = k - 1
    if done == 0:
        if x > 0:
            if k == 0:
                w = x
                done = 1
            k = k - 1
    if done == 0:
        if k < nmain:
            w = F
            done = 1
        k = k - nmain
    if done == 0:
        if t1 > 0:
            if k == 0:
                w = t1
                done = 1
            k = k - 1
    if done == 0:
        if t2 > 0:
            if k == 0:
                w = t2
                done = 1
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
def interleave_kernel(A, B, rows, f_max, f_min, f_align, nb_min, w_min, r_min):
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

    L, tail, F, h, x, nmain, t1, t2, nb, D = _plan(cnt, rows, f_max, f_min, f_align,
                                                   nb_min, w_min, r_min)

    # single-partition tail: loads issued first; zip/store placed after block 0
    toff = start + rows * L
    ta = A
    tb = B
    if tail > 0:
        ta = _load(A, toff, 1, tail, dt)
        tb = _load(B, toff, 1, tail, dt)
        if nb == 0:
            _zip_store(out, ta, tb, toff, 1, tail, dt, True)

    if nb > 0:
        lo = start
        w0 = _bw(0, h, x, nmain, F, t1, t2)
        a0 = _load(A, lo, rows, w0, dt)
        b0 = _load(B, lo, rows, w0, dt)
        lo = lo + rows * w0
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
                wi = _bw(i, h, x, nmain, F, t1, t2)
                pa = _load(A, lo, rows, wi, dt)
                pb = _load(B, lo, rows, wi, dt)
                lo = lo + rows * wi
                if i == 1:
                    a1 = pa
                    b1 = pb
                if i == 2:
                    a2 = pa
                    b2 = pb
                if i == 3:
                    a3 = pa
                    b3 = pb
        zo = start
        for j in range(nb):
            if j + D < nb:
                wn = _bw(j + D, h, x, nmain, F, t1, t2)
                na = _load(A, lo, rows, wn, dt)
                nbt = _load(B, lo, rows, wn, dt)
                lo = lo + rows * wn
                if D == 2:
                    a2 = na
                    b2 = nbt
                if D == 3:
                    a3 = na
                    b3 = nbt
                if D == 4:
                    a4 = na
                    b4 = nbt
            wj = _bw(j, h, x, nmain, F, t1, t2)
            use_vec = (j % 2) == 0
            _zip_store(out, a0, b0, zo, rows, wj, dt, use_vec)
            zo = zo + rows * wj
            if j == 0:
                if tail > 0:
                    _zip_store(out, ta, tb, toff, 1, tail, dt, True)
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
        L, tail, F, h, x, nmain, t1, t2, nb, D = _plan(cnt, ROWS, F_MAX, F_MIN, F_ALIGN,
                                                       NB_MIN, W_MIN, R_MIN)
        cores.append({
            "start": start,
            "count": cnt,
            "rows": ROWS,
            "elems_per_partition": L,
            "F": F,
            "head_F": h,
            "extra_F": x,
            "main_blocks": nmain,
            "taper1_F": t1,
            "taper2_F": t2,
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
        "w_min": W_MIN,
        "r_min": R_MIN,
        "programs": N_PROGRAMS,
        "zip_engines": "alternate vector/scalar per block",
        "tail_placement": "loads first, zip/store after block 0",
        "cores": cores,
    }


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(A.shape[0])
    out = interleave_kernel(A, B, ROWS, F_MAX, F_MIN, F_ALIGN, NB_MIN, W_MIN, R_MIN)
    _LAST_CONFIG = _config_for(n)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
