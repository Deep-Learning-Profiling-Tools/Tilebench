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
F_ALIGN = 256      # main block width multiple
NB_MIN = 12        # aim for at least this many main blocks per core
N_PROGRAMS = 2     # LNC=2: two physical cores per logical core
SO = 3             # number of rotating output slots

_LAST_CONFIG = {}


def _plan(C, rows, f_max, f_min, f_align, nb_min):
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
    Q = F // 4
    if Q < 1:
        Q = 1
    q0 = 0                        # small head block
    if F > 0:
        if L > 2 * F:
            q0 = Q
    rem = L - q0
    nmain = 0
    flast = 0
    if F > 0:
        nmain = rem // F
        flast = rem - nmain * F
    R = flast
    if flast == 0:
        if nmain > 0:
            nmain = nmain - 1
            R = F
    wa = 0                        # penultimate block
    wb = 0                        # small final block
    if R > 0:
        if R <= Q:
            wb = R
        elif R <= 2 * Q:
            wb = R // 2
            wa = R - wb
        else:
            wb = Q
            wa = R - Q
    nb = nmain
    if q0 > 0:
        nb = nb + 1
    if wa > 0:
        nb = nb + 1
    if wb > 0:
        nb = nb + 1
    return L, tail, F, q0, nmain, wa, wb, nb


def _slots(F):
    S = 4
    if F <= 2048:
        S = 5
    if F <= 1024:
        S = 6
    return S


def _bw(j, q0, nmain, F, wa, wb):
    h = 0
    if q0 > 0:
        h = 1
    w = wb
    if j < h:
        w = q0
    elif j < h + nmain:
        w = F
    else:
        k = j - h - nmain
        if wa > 0:
            if k == 0:
                w = wa
    return w


def _bo(j, q0, nmain, F, wa, wb):
    off = 0
    for i in range(j):
        off = off + _bw(i, q0, nmain, F, wa, wb)
    return off


def _pick6(k, t0, t1, t2, t3, t4, t5):
    r = t0
    if k == 1:
        r = t1
    if k == 2:
        r = t2
    if k == 3:
        r = t3
    if k == 4:
        r = t4
    if k == 5:
        r = t5
    return r


def _pick3(k, t0, t1, t2):
    r = t0
    if k == 1:
        r = t1
    if k == 2:
        r = t2
    return r


def _load_slot(X, t, off, rows, w):
    # contiguous X[off : off + rows*w] -> slot t[:rows, :w]
    nisa.dma_copy(dst=t[:, 0:w], src=X.ap(pattern=[[w, rows], [1, w]], offset=off))


def _zip_store(O, ta, tb, to, off, rows, w):
    # to[p, 2k] = ta[p, k] (VectorE), to[p, 2k+1] = tb[p, k] (ScalarE)
    nisa.tensor_copy(dst=to[:, 0:2 * w:2], src=ta[:, 0:w], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=to[:, 1:2 * w:2], src=tb[:, 0:w], engine=nisa.engine.scalar)
    nisa.dma_copy(dst=O.ap(pattern=[[2 * w, rows], [1, 2 * w]], offset=2 * off),
                  src=to[:, 0:2 * w])


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

    L, tail, F, q0, nmain, wa, wb, nb = _plan(cnt, rows, f_max, f_min, f_align, nb_min)

    # small single-partition tail (independent; issued first so it overlaps)
    if tail > 0:
        toff = start + rows * L
        ta = nl.ndarray((1, tail), dtype=dt, buffer=nl.sbuf)
        tb = nl.ndarray((1, tail), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=ta, src=A.ap(pattern=[[tail, 1], [1, tail]], offset=toff))
        nisa.dma_copy(dst=tb, src=B.ap(pattern=[[tail, 1], [1, tail]], offset=toff))
        to = nl.ndarray((1, 2 * tail), dtype=dt, buffer=nl.sbuf)
        nisa.tensor_copy(dst=to[:, 0:2 * tail:2], src=ta, engine=nisa.engine.vector)
        nisa.tensor_copy(dst=to[:, 1:2 * tail:2], src=tb, engine=nisa.engine.vector)
        nisa.dma_copy(dst=out.ap(pattern=[[2 * tail, 1], [1, 2 * tail]], offset=2 * toff),
                      src=to)

    if nb > 0:
        S = _slots(F)
        D = S - 1

        # explicit rotating slot buffers (allocated once)
        a0 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        a1 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        a2 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        a3 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        b0 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        b1 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        b2 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        b3 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        a4 = a0
        b4 = b0
        a5 = a0
        b5 = b0
        if S > 4:
            a4 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
            b4 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        if S > 5:
            a5 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
            b5 = nl.ndarray((rows, F), dtype=dt, buffer=nl.sbuf)
        o0 = nl.ndarray((rows, 2 * F), dtype=dt, buffer=nl.sbuf)
        o1 = nl.ndarray((rows, 2 * F), dtype=dt, buffer=nl.sbuf)
        o2 = nl.ndarray((rows, 2 * F), dtype=dt, buffer=nl.sbuf)

        # prologue: loads for blocks 0 .. D-1
        for i in range(D):
            if i < nb:
                wi = _bw(i, q0, nmain, F, wa, wb)
                offi = start + rows * _bo(i, q0, nmain, F, wa, wb)
                ki = i % S
                _load_slot(A, _pick6(ki, a0, a1, a2, a3, a4, a5), offi, rows, wi)
                _load_slot(B, _pick6(ki, b0, b1, b2, b3, b4, b5), offi, rows, wi)

        for j in range(nb):
            if j + D < nb:
                jn = j + D
                wn = _bw(jn, q0, nmain, F, wa, wb)
                offn = start + rows * _bo(jn, q0, nmain, F, wa, wb)
                kn = jn % S
                _load_slot(A, _pick6(kn, a0, a1, a2, a3, a4, a5), offn, rows, wn)
                _load_slot(B, _pick6(kn, b0, b1, b2, b3, b4, b5), offn, rows, wn)
            wj = _bw(j, q0, nmain, F, wa, wb)
            offj = start + rows * _bo(j, q0, nmain, F, wa, wb)
            kj = j % S
            ko = j % SO
            _zip_store(out,
                       _pick6(kj, a0, a1, a2, a3, a4, a5),
                       _pick6(kj, b0, b1, b2, b3, b4, b5),
                       _pick3(ko, o0, o1, o2),
                       offj, rows, wj)
    return out


def _config_for(n):
    per = n // N_PROGRAMS
    cores = []
    for pid in range(N_PROGRAMS):
        start = pid * per
        cnt = per if pid < N_PROGRAMS - 1 else n - start
        L, tail, F, q0, nmain, wa, wb, nb = _plan(cnt, ROWS, F_MAX, F_MIN, F_ALIGN, NB_MIN)
        S = _slots(F) if nb > 0 else 0
        cores.append({
            "start": start,
            "count": cnt,
            "rows": ROWS,
            "elems_per_partition": L,
            "F": F,
            "head_block_F": q0,
            "main_blocks": nmain,
            "penultimate_F": wa,
            "final_F": wb,
            "blocks": nb,
            "load_slots": S,
            "prefetch": (S - 1) if S > 0 else 0,
            "out_slots": SO,
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
        "zip_engines": "A->vector, B->scalar per block",
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
