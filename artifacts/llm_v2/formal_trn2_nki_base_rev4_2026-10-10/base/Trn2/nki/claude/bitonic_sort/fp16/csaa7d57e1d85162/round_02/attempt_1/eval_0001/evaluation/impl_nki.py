import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
_INF = float("inf")
_P2 = tuple(1 << i for i in range(48))
_LAST_CONFIG = {}


def _mm(dst, a, b, is_min):
    if is_min:
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=nl.minimum)
    else:
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=nl.maximum)


def _cx(sv, dv, tv, p0, p1, a0, a1, b0, b1, c0, c1, asc, pp):
    lo = sv[p0:p1, a0:a1, b0:b1, 0, c0:c1]
    hi = sv[p0:p1, a0:a1, b0:b1, 1, c0:c1]
    nasc = True
    if asc:
        nasc = False
    if pp:
        _mm(dv[p0:p1, a0:a1, b0:b1, 0, c0:c1], lo, hi, asc)
        _mm(dv[p0:p1, a0:a1, b0:b1, 1, c0:c1], lo, hi, nasc)
    else:
        t = tv[p0:p1, a0:a1, b0:b1, c0:c1]
        _mm(t, lo, hi, asc)
        _mm(hi, lo, hi, nasc)
        nisa.tensor_copy(dst=lo, src=t, engine=nisa.engine.scalar)


def _region(src, dst, tmp, p0, p1, A, NB, Bm, boff, C, asc, pp, CH):
    sv = src.reshape((P, A, NB, 2, C))
    dv = sv
    tv = sv
    if pp:
        dv = dst.reshape((P, A, NB, 2, C))
    else:
        tv = tmp.reshape((P, A, NB, C))
    BC = Bm * C
    if BC <= CH:
        ac = min(A, CH // BC)
        for a0 in range(0, A, ac):
            _cx(sv, dv, tv, p0, p1, a0, a0 + ac, boff, boff + Bm, 0, C, asc, pp)
    elif C <= CH:
        bc = CH // C
        for a in range(A):
            for b0 in range(boff, boff + Bm, bc):
                _cx(sv, dv, tv, p0, p1, a, a + 1, b0, b0 + bc, 0, C, asc, pp)
    else:
        for a in range(A):
            for b in range(boff, boff + Bm):
                for c0 in range(0, C, CH):
                    _cx(sv, dv, tv, p0, p1, a, a + 1, b, b + 1, c0, c0 + CH, asc, pp)


def _stage(cur, oth, tmp, F, jb, db, mode, asc, pp, CH):
    # mode 1: direction = free bit db; mode 0: constant asc; mode 2: partition halves
    C = _P2[jb]
    if mode == 1:
        A = F // _P2[db + 1]
        Bm = _P2[db - jb - 1]
        _region(cur, oth, tmp, 0, P, A, 2 * Bm, Bm, 0, C, True, pp, CH)
        _region(cur, oth, tmp, 0, P, A, 2 * Bm, Bm, Bm, C, False, pp, CH)
    elif mode == 0:
        _region(cur, oth, tmp, 0, P, F // _P2[jb + 1], 1, 1, 0, C, asc, pp, CH)
    else:
        _region(cur, oth, tmp, 0, 64, F // _P2[jb + 1], 1, 1, 0, C, True, pp, CH)
        _region(cur, oth, tmp, 64, P, F // _P2[jb + 1], 1, 1, 0, C, False, pp, CH)


def _xfer_a(X, work, base, F, r, store):
    # partition q = bit*64 + a*B + b  <->  segment row h = a*2B + bit*B + b
    B = _P2[r]
    A = 64 // B
    FC = min(F, 32768)
    for bit in range(2):
        for a in range(A):
            q0 = bit * 64 + a * B
            h0 = a * 2 * B + bit * B
            for c0 in range(0, F, FC):
                hv = work.ap(pattern=[[F, B], [1, FC]], offset=base + h0 * F + c0)
                if store:
                    nisa.dma_copy(dst=hv, src=X[q0:q0 + B, c0:c0 + FC])
                else:
                    nisa.dma_copy(dst=X[q0:q0 + B, c0:c0 + FC], src=hv)


def _xfer_b(X, work, base, F, store):
    # X[p, h*W + l] = seg[h*F + p*W + l]
    W = F // P
    X3 = X.reshape((P, P, W))
    HC = 8
    for h0 in range(0, P, HC):
        hv = work.ap(pattern=[[W, P], [F, HC], [1, W]], offset=base + h0 * F)
        if store:
            nisa.dma_copy(dst=hv, src=X3[:, h0:h0 + HC, :])
        else:
            nisa.dma_copy(dst=X3[:, h0:h0 + HC, :], src=hv)


@nki.jit
def _bitonic_kernel(data, L, pp, CH):
    N = data.shape[0]
    M = _P2[L]
    work = nl.ndarray((M,), dtype=data.dtype, buffer=nl.shared_hbm)
    c = nl.program_id(0)
    Ls = L - 1
    S = _P2[Ls]
    Lf = Ls - 7
    F = _P2[Lf]
    w = Lf - 7
    base = c * S
    X = nl.ndarray((P, F), dtype=data.dtype, buffer=nl.sbuf)
    if pp:
        Y = nl.ndarray((P, F), dtype=data.dtype, buffer=nl.sbuf)
        T = Y
    else:
        T = nl.ndarray((P, F // 2), dtype=data.dtype, buffer=nl.sbuf)
        Y = T
    # pad pass fused into the first local group
    MC = min(F, 8192)
    for c0 in range(0, F, MC):
        nisa.memset(dst=X[:, c0:c0 + MC], value=_INF)
    nv = N - base
    if nv > S:
        nv = S
    if nv < 0:
        nv = 0
    rows = nv // F
    rem = nv - rows * F
    FC = min(F, 32768)
    for g0 in range(0, rows, 32):
        gs = min(32, rows - g0)
        for c0 in range(0, F, FC):
            nisa.dma_copy(dst=X[g0:g0 + gs, c0:c0 + FC],
                          src=data.ap(pattern=[[F, gs], [1, FC]], offset=base + g0 * F + c0))
    for c0 in range(0, rem, FC):
        cs = min(FC, rem - c0)
        nisa.dma_copy(dst=X[rows:rows + 1, c0:c0 + cs],
                      src=data.ap(pattern=[[F, 1], [1, cs]], offset=base + rows * F + c0))
    cur = X
    oth = Y
    kind = 0
    rr = 6
    for m in range(1, L + 1):
        jb = m - 1
        asc = True
        if m == Ls:
            asc = c == 0
        if m == L:
            # global stage (M, M/2): both halves in work, synchronise, combine
            _xfer_a(cur, work, base, F, rr, True)
            nisa.core_barrier(data=work, cores=(0, 1))
            ob = (1 - c) * S
            sb = T
            SC = F // 2
            if pp:
                sb = oth
                SC = F
            FC2 = min(SC, 32768)
            for h0 in range(0, F, SC):
                for g in range(2):
                    for c0 in range(0, SC, FC2):
                        nisa.dma_copy(dst=sb[g * 64:(g + 1) * 64, c0:c0 + FC2],
                                      src=work.ap(pattern=[[F, 64], [1, FC2]],
                                                  offset=ob + g * 64 * F + h0 + c0))
                for c0 in range(0, SC, CH):
                    cs = min(CH, SC - c0)
                    _mm(cur[:, h0 + c0:h0 + c0 + cs], cur[:, h0 + c0:h0 + c0 + cs],
                        sb[:, c0:c0 + cs], c == 0)
            nisa.core_barrier(data=work, cores=(0, 1))
            jb = Ls - 1
        if jb >= Lf:
            if kind == 0:
                _xfer_a(cur, work, base, F, rr, True)
                _xfer_b(cur, work, base, F, False)
                kind = 1
            while jb >= Lf:
                if m < Ls:
                    _stage(cur, oth, T, F, jb - Lf + w, m - Lf + w, 1, asc, pp, CH)
                else:
                    _stage(cur, oth, T, F, jb - Lf + w, 0, 0, asc, pp, CH)
                if pp:
                    tt = cur
                    cur = oth
                    oth = tt
                jb = jb - 1
        nr = 6
        mode = 0
        if m < Lf:
            mode = 1
        elif m < Ls:
            mode = 2
            nr = m - Lf
        if kind == 1:
            _xfer_b(cur, work, base, F, True)
            _xfer_a(cur, work, base, F, nr, False)
            kind = 0
            rr = nr
        elif rr != nr:
            _xfer_a(cur, work, base, F, rr, True)
            _xfer_a(cur, work, base, F, nr, False)
            rr = nr
        while jb >= 0:
            _stage(cur, oth, T, F, jb, m, mode, asc, pp, CH)
            if pp:
                tt = cur
                cur = oth
                oth = tt
            jb = jb - 1
    _xfer_a(cur, work, base, F, rr, True)
    return work


def run(data: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    if N <= 1:
        return data.clone()
    M = 1 << ((N - 1).bit_length())
    L = M.bit_length() - 1
    F = 1 << (L - 8)
    pp = F <= 32768
    CH = 16384 if pp else 8192
    work = _bitonic_kernel[2](data, L, pp, CH)
    _LAST_CONFIG = {
        "M": M,
        "L": L,
        "programs": 2,
        "free_per_partition": F,
        "pingpong": bool(pp),
        "chunk": CH,
    }
    return work[:N]


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
