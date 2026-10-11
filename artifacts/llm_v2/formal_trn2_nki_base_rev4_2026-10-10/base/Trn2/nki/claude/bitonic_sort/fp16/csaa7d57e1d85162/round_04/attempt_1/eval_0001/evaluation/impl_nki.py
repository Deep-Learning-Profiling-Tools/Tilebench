import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
_INF = float("inf")
_P2 = tuple(1 << i for i in range(31))
_CH = 8192      # max pairs per partition per compare instruction
_CC = 4096      # columns per chunk in the cross-core stage
_MC = 8192      # columns per memset instruction
_LAST_CONFIG = {}


def _mm(dst, a, b, is_min):
    if is_min:
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=nl.minimum,
                           engine=nisa.engine.vector)
    else:
        nisa.tensor_tensor(dst=dst, data1=a, data2=b, op=nl.maximum,
                           engine=nisa.engine.vector)


def _cx4(sv, dv, a0, a1, c0, c1, asc):
    nasc = True
    if asc:
        nasc = False
    lo = sv[0:P, a0:a1, 0, c0:c1]
    hi = sv[0:P, a0:a1, 1, c0:c1]
    _mm(dv[0:P, a0:a1, 0, c0:c1], lo, hi, asc)
    _mm(dv[0:P, a0:a1, 1, c0:c1], lo, hi, nasc)


def _cx5(sv, dv, a0, a1, b0, b1, c0, c1, asc):
    nasc = True
    if asc:
        nasc = False
    lo = sv[0:P, a0:a1, b0:b1, 0, c0:c1]
    hi = sv[0:P, a0:a1, b0:b1, 1, c0:c1]
    _mm(dv[0:P, a0:a1, b0:b1, 0, c0:c1], lo, hi, asc)
    _mm(dv[0:P, a0:a1, b0:b1, 1, c0:c1], lo, hi, nasc)


def _ureg(src, dst, Fh, jp, asc, CH):
    # uniform direction; partner distance 2^jp inside a half buffer
    C = _P2[jp]
    A = Fh // (2 * C)
    sv = src.reshape((P, A, 2, C))
    dv = dst.reshape((P, A, 2, C))
    if C <= CH:
        ac = CH // C
        if ac > A:
            ac = A
        for a0 in range(0, A, ac):
            _cx4(sv, dv, a0, a0 + ac, 0, C, asc)
    else:
        for a in range(A):
            for c0 in range(0, C, CH):
                _cx4(sv, dv, a, a + 1, c0, c0 + CH, asc)


def _freg(src, dst, Fh, jp, dp, CH):
    # direction given by free bit dp (> jp) inside a half buffer
    C = _P2[jp]
    Bm = _P2[dp - jp - 1]
    A = Fh // _P2[dp + 1]
    sv = src.reshape((P, A, 2 * Bm, 2, C))
    dv = dst.reshape((P, A, 2 * Bm, 2, C))
    BC = Bm * C
    for d in range(2):
        asc = d == 0
        boff = d * Bm
        if BC <= CH:
            ac = CH // BC
            if ac > A:
                ac = A
            for a0 in range(0, A, ac):
                _cx5(sv, dv, a0, a0 + ac, boff, boff + Bm, 0, C, asc)
        elif C <= CH:
            bc = CH // C
            for a in range(A):
                for b0 in range(boff, boff + Bm, bc):
                    _cx5(sv, dv, a, a + 1, b0, b0 + bc, 0, C, asc)
        else:
            for a in range(A):
                for b in range(boff, boff + Bm):
                    for c0 in range(0, C, CH):
                        _cx5(sv, dv, a, a + 1, b, b + 1, c0, c0 + CH, asc)


def _xstage(h0, h1, fr, Fh, asc, CH):
    # partner = other half buffer (elementwise); new low -> fr, new high -> h1 in place
    nasc = True
    if asc:
        nasc = False
    for c0 in range(0, Fh, CH):
        c1 = min(Fh, c0 + CH)
        a = h0[0:P, c0:c1]
        b = h1[0:P, c0:c1]
        _mm(fr[0:P, c0:c1], a, b, asc)
        _mm(b, a, b, nasc)


def _stage(h0, h1, fr, Fh, Lf, jp, dk, dp, asc, CH):
    # dk: 0 constant direction (asc), 1 free bit dp, 3 direction = half index
    if jp == Lf - 1:
        _xstage(h0, h1, fr, Fh, asc, CH)
    elif dk == 3:
        _ureg(h0, fr, Fh, jp, True, CH)
        _ureg(h1, h0, Fh, jp, False, CH)
    elif dk == 1:
        _freg(h0, fr, Fh, jp, dp, CH)
        _freg(h1, h0, Fh, jp, dp, CH)
    else:
        _ureg(h0, fr, Fh, jp, asc, CH)
        _ureg(h1, h0, Fh, jp, asc, CH)


def _xfer(lay, lm, h0, h1, work, c, Ls, Lf, store):
    # move this core's segment between SBUF (layout `lay`) and its region of work
    Fh = _P2[Lf - 1]
    W = _P2[Lf - 7]
    if lay == 0:
        # A: partition = e[Lf..Ls-1], free = e[0..Lf-1] (half bit e[Lf-1])
        WA = work.reshape((2 * P, 2, Fh))
        for q0 in range(0, P, 32):
            r0 = c * P + q0
            if store:
                nisa.dma_copy(dst=WA[r0:r0 + 32, 0, :], src=h0[q0:q0 + 32, :])
                nisa.dma_copy(dst=WA[r0:r0 + 32, 1, :], src=h1[q0:q0 + 32, :])
            else:
                nisa.dma_copy(dst=h0[q0:q0 + 32, :], src=WA[r0:r0 + 32, 0, :])
                nisa.dma_copy(dst=h1[q0:q0 + 32, :], src=WA[r0:r0 + 32, 1, :])
    elif lay == 1:
        # B: partition = e[w..Lf-1], free = e[0..w-1] + e[Lf..Ls-1] (half bit e[Ls-1])
        WB = work.reshape((2 * P, P, W))
        for h in range(P):
            t = h0
            hh = h
            if h >= 64:
                t = h1
                hh = h - 64
            if store:
                nisa.dma_copy(dst=WB[c * P + h], src=t[0:P, hh * W:(hh + 1) * W])
            else:
                nisa.dma_copy(dst=t[0:P, hh * W:(hh + 1) * W], src=WB[c * P + h])
    elif lay == 2:
        # X1: partition = e[w-1..Lf-2], free = e[0..w-2] + e[Lf-1..Ls-1] (half bit e[Ls-1])
        Wh = W // 2
        WX = work.reshape((4 * P, P, Wh))
        for h in range(2 * P):
            t = h0
            hh = h
            if h >= P:
                t = h1
                hh = h - P
            if store:
                nisa.dma_copy(dst=WX[c * 2 * P + h], src=t[0:P, hh * Wh:(hh + 1) * Wh])
            else:
                nisa.dma_copy(dst=t[0:P, hh * Wh:(hh + 1) * Wh], src=WX[c * 2 * P + h])
    else:
        # X2_lm: partition = e[Lf-1..Ls-1] minus e[lm], free = e[0..Lf-2], half bit = e[lm]
        nlo = _P2[lm - Lf + 1]
        nhi = _P2[Ls - 1 - lm]
        WX2 = work.reshape((2 * nhi, 2, nlo, Fh))
        for ph in range(nhi):
            q0 = ph * nlo
            r = c * nhi + ph
            if store:
                nisa.dma_copy(dst=WX2[r, 0], src=h0[q0:q0 + nlo, :])
                nisa.dma_copy(dst=WX2[r, 1], src=h1[q0:q0 + nlo, :])
            else:
                nisa.dma_copy(dst=h0[q0:q0 + nlo, :], src=WX2[r, 0])
                nisa.dma_copy(dst=h1[q0:q0 + nlo, :], src=WX2[r, 1])


@nki.jit
def _pad_sort_kernel(data, work, L, CH):
    # pad + all stages k <= 2^(Lf-1) (partner inside a partition row), layout A
    N = data.shape[0]
    dt = data.dtype
    c = nl.program_id(0)
    Ls = L - 1
    Lf = Ls - 7
    S = _P2[Ls]
    Fh = _P2[Lf - 1]
    ba = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    bb = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    bf = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    h0 = ba
    h1 = bb
    fr = bf
    D2 = data.reshape((1, N))
    base = c * S
    nv = N - base
    if nv > S:
        nv = S
    if nv < 0:
        nv = 0
    nfull = nv // Fh
    rem = nv - nfull * Fh
    n0 = (nfull + 1) // 2
    n1 = nfull // 2
    if n0 < P:
        pa = (n0 // 32) * 32
        for c0 in range(0, Fh, _MC):
            c1 = min(Fh, c0 + _MC)
            nisa.memset(dst=h0[pa:P, c0:c1], value=_INF)
    if n1 < P:
        pb = (n1 // 32) * 32
        for c0 in range(0, Fh, _MC):
            c1 = min(Fh, c0 + _MC)
            nisa.memset(dst=h1[pb:P, c0:c1], value=_INF)
    for p in range(n0):
        off = base + 2 * p * Fh
        nisa.dma_copy(dst=h0[p:p + 1, :], src=D2[0:1, off:off + Fh])
    for p in range(n1):
        off = base + (2 * p + 1) * Fh
        nisa.dma_copy(dst=h1[p:p + 1, :], src=D2[0:1, off:off + Fh])
    if rem > 0:
        off = base + nfull * Fh
        pr = nfull // 2
        if nfull - 2 * pr == 0:
            nisa.dma_copy(dst=h0[pr:pr + 1, 0:rem], src=D2[0:1, off:off + rem])
        else:
            nisa.dma_copy(dst=h1[pr:pr + 1, 0:rem], src=D2[0:1, off:off + rem])
    for m in range(1, Lf):
        jb = m - 1
        while jb >= 0:
            dk = 1
            if m == Lf - 1:
                dk = 3
            _stage(h0, h1, fr, Fh, Lf, jb, dk, m, True, CH)
            tt = h1
            h1 = h0
            h0 = fr
            fr = tt
            jb = jb - 1
    _xfer(0, 0, h0, h1, work, c, Ls, Lf, True)
    res = nl.ndarray((2, 32), dtype=nl.float32, buffer=nl.shared_hbm)
    zt = nl.ndarray((1, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zt, value=0.0)
    nisa.dma_copy(dst=res[c:c + 1, :], src=zt)
    return res


@nki.jit
def _local_merge_kernel(work, L, CH):
    # stages k = 2^Lf .. 2^Ls on each core's half (core 0 asc, core 1 desc at k = 2^Ls)
    dt = work.dtype
    c = nl.program_id(0)
    Ls = L - 1
    Lf = Ls - 7
    Fh = _P2[Lf - 1]
    ba = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    bb = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    bf = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    h0 = ba
    h1 = bb
    fr = bf
    lay = -1
    lm = 0
    for m in range(Lf, Ls + 1):
        jb = m - 1
        while jb >= 0:
            if m < Ls:
                need = 3
                if jb >= Lf - 1:
                    need = 2
            else:
                need = 0
                if jb >= Lf:
                    need = 1
            sw = need != lay
            if need == 3:
                if lm != m:
                    sw = True
            if sw:
                if lay >= 0:
                    _xfer(lay, lm, h0, h1, work, c, Ls, Lf, True)
                _xfer(need, m, h0, h1, work, c, Ls, Lf, False)
                lay = need
                lm = m
            if lay == 3:
                jp = jb
                dp = Lf - 1
            elif lay == 0:
                jp = jb
                dp = m
            else:
                jp = jb - 7
                dp = m - 7
            asc = True
            dk = 1
            if m == Ls:
                dk = 0
                asc = c == 0
            elif dp == Lf - 1:
                dk = 3
            _stage(h0, h1, fr, Fh, Lf, jp, dk, dp, asc, CH)
            if jp == Lf - 1:
                tt = h0
                h0 = fr
                fr = tt
            else:
                tt = h1
                h1 = h0
                h0 = fr
                fr = tt
            jb = jb - 1
    _xfer(lay, lm, h0, h1, work, c, Ls, Lf, True)
    res = nl.ndarray((2, 32), dtype=nl.float32, buffer=nl.shared_hbm)
    zt = nl.ndarray((1, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zt, value=0.0)
    nisa.dma_copy(dst=res[c:c + 1, :], src=zt)
    return res


@nki.jit
def _cross_core_kernel(work, L, CC):
    # stage (k=M, j=M/2): core c owns pairs (i, i+M/2) for i in quarter c
    dt = work.dtype
    c = nl.program_id(0)
    Ls = L - 1
    Lf = Ls - 7
    Fh = _P2[Lf - 1]
    W4 = work.reshape((2, 2, P, Fh))
    cc = min(CC, Fh)
    ta = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    tb = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    for c0 in range(0, Fh, cc):
        nisa.dma_copy(dst=ta[0:P, c0:c0 + cc], src=W4[0, c, 0:P, c0:c0 + cc])
        nisa.dma_copy(dst=tb[0:P, c0:c0 + cc], src=W4[1, c, 0:P, c0:c0 + cc])
    for c0 in range(0, Fh, cc):
        tm = nl.ndarray((P, cc), dtype=dt, buffer=nl.sbuf)
        a = ta[0:P, c0:c0 + cc]
        b = tb[0:P, c0:c0 + cc]
        _mm(tm, a, b, True)
        _mm(b, a, b, False)
        nisa.dma_copy(dst=W4[0, c, 0:P, c0:c0 + cc], src=tm)
        nisa.dma_copy(dst=W4[1, c, 0:P, c0:c0 + cc], src=b)
    res = nl.ndarray((2, 32), dtype=nl.float32, buffer=nl.shared_hbm)
    zt = nl.ndarray((1, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zt, value=0.0)
    nisa.dma_copy(dst=res[c:c + 1, :], src=zt)
    return res


@nki.jit
def _final_merge_kernel(work, L, CH):
    # stages (k=M, j=M/4 .. 1), ascending, on each core's half
    dt = work.dtype
    c = nl.program_id(0)
    Ls = L - 1
    Lf = Ls - 7
    Fh = _P2[Lf - 1]
    ba = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    bb = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    bf = nl.ndarray((P, Fh), dtype=dt, buffer=nl.sbuf)
    h0 = ba
    h1 = bb
    fr = bf
    _xfer(1, 0, h0, h1, work, c, Ls, Lf, False)
    lay = 1
    jb = Ls - 1
    while jb >= 0:
        if lay == 1:
            if jb < Lf:
                _xfer(1, 0, h0, h1, work, c, Ls, Lf, True)
                _xfer(0, 0, h0, h1, work, c, Ls, Lf, False)
                lay = 0
        if lay == 1:
            jp = jb - 7
        else:
            jp = jb
        _stage(h0, h1, fr, Fh, Lf, jp, 0, 0, True, CH)
        if jp == Lf - 1:
            tt = h0
            h0 = fr
            fr = tt
        else:
            tt = h1
            h1 = h0
            h0 = fr
            fr = tt
        jb = jb - 1
    _xfer(lay, 0, h0, h1, work, c, Ls, Lf, True)
    res = nl.ndarray((2, 32), dtype=nl.float32, buffer=nl.shared_hbm)
    zt = nl.ndarray((1, 32), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zt, value=0.0)
    nisa.dma_copy(dst=res[c:c + 1, :], src=zt)
    return res


def run(data: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    if N <= 1:
        return data.clone()
    M = 1 << ((N - 1).bit_length())
    L = M.bit_length() - 1
    if L < 16 or L > 24:
        raise ValueError("bitonic_sort kernel supports 32768 < N <= 16777216")
    work = torch.empty((M,), dtype=data.dtype, device=data.device)
    _pad_sort_kernel[2](data, work, L, _CH)
    _local_merge_kernel[2](work, L, _CH)
    _cross_core_kernel[2](work, L, _CC)
    _final_merge_kernel[2](work, L, _CH)
    _LAST_CONFIG = {
        "M": M,
        "L": L,
        "programs": 2,
        "kernels": 4,
        "free_per_partition": 1 << (L - 8),
        "half_buffers": 3,
        "compare_chunk": _CH,
        "cross_chunk": _CC,
        "memset_chunk": _MC,
    }
    return work[:N]


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
