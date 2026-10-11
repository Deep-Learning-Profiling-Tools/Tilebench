import math
import torch
import nki
import nki.language as nl
import nki.isa as nisa

# LayerNorm (fp16 in/out, fp32 statistics + apply) on Trn2, LNC=2.
# Rows viewed as (2*128 partitions, rpp*K); each partition owns rpp rows.
# Per group of R rows/partition:
#   DMA load -> stats (ScalarE sum & sumsq, fp32; group 0: VectorE sum)
#   -> tiny fp32 finalize (VectorE + GpSimd pow + Newton)
#   -> VectorE STT: u=(x-mean)*W (fp32), y=u*rstd+B (fp16) -> DMA store.
# Stats run two groups ahead of apply, finalize one group ahead.

_P = 128
_LAST_CONFIG = {}
_PATH_CACHE = {}
_PATH_NAMES = ("stt_vector", "stt_default", "act_vector", "r1_fallback")


def _pick3(i, a, b, c):
    r = c
    if i == 0:
        r = a
    elif i == 1:
        r = b
    return r


def _pick4(i, a, b, c, d):
    r = d
    if i == 0:
        r = a
    elif i == 1:
        r = b
    elif i == 2:
        r = c
    return r


def _chunk_lo(c, nc, nblk, k):
    v = ((c * nblk) // nc) * 512
    if v > k:
        v = k
    return v


def _ld_group(xa, xr, p0, base, R, NC, nblk, K):
    for j in range(R):
        for c in range(NC):
            a0 = _chunk_lo(c, NC, nblk, K)
            a1 = _chunk_lo(c + 1, NC, nblk, K)
            o = j * K + a0
            nisa.dma_copy(dst=xa[:, o:o + a1 - a0], src=xr[p0:p0 + 128, base + o:base + o + a1 - a0])


def _st_group(xs, st, junk, R, NC, nblk, K, vec_s):
    NI = R * NC
    for j in range(R):
        for c in range(NC):
            a0 = _chunk_lo(c, NC, nblk, K)
            a1 = _chunk_lo(c + 1, NC, nblk, K)
            cw = a1 - a0
            o = j * K + a0
            col = j * NC + c
            if vec_s == 1:
                nisa.tensor_reduce(dst=st[:, col:col + 1], op=nl.add, data=xs[:, o:o + cw], axis=1)
            else:
                nisa.activation(dst=junk[:, 0:cw], op=nl.copy, data=xs[:, o:o + cw],
                                reduce_op=nl.add, reduce_res=st[:, col:col + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            nisa.activation(dst=junk[:, 0:cw], op=nl.square, data=xs[:, o:o + cw],
                            reduce_op=nl.add, reduce_res=st[:, NI + col:NI + col + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)


def _fin_a(st, mhalf, R, NC, K, keps):
    NI = R * NC
    a1 = 0
    a2 = NI
    if NC > 1:
        a1 = 2 * NI
        a2 = 2 * NI + R
        for j in range(R):
            nisa.tensor_reduce(dst=st[:, a1 + j:a1 + j + 1], op=nl.add,
                               data=st[:, j * NC:(j + 1) * NC], axis=1)
            nisa.tensor_reduce(dst=st[:, a2 + j:a2 + j + 1], op=nl.add,
                               data=st[:, NI + j * NC:NI + (j + 1) * NC], axis=1)
    on = 2 * NI + 2 * R
    ov = on + R
    o0 = ov + R
    # nm = -S/K
    nisa.tensor_scalar(dst=st[:, on:on + R], data=st[:, a1:a1 + R], op0=nl.multiply,
                       operand0=-1.0 / K)
    # K*var = Q - S^2/K
    tq = nl.ndarray((128, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tq, data1=st[:, a1:a1 + R], data2=st[:, on:on + R], op=nl.multiply)
    nisa.tensor_tensor(dst=st[:, ov:ov + R], data1=st[:, a2:a2 + R], data2=tq, op=nl.add)
    # vk = max(K*var, 0) + K*eps
    nisa.tensor_scalar(dst=st[:, ov:ov + R], data=st[:, ov:ov + R], op0=nl.maximum,
                       operand0=0.0, op1=nl.add, operand1=keps)
    # r0 ~ vk^-0.5
    nisa.tensor_tensor(dst=st[:, o0:o0 + R], data1=st[:, ov:ov + R], data2=mhalf,
                       op=nl.power, engine=nisa.engine.gpsimd)


def _fin_b(st, R, NC, sk, need_nmr):
    NI = R * NC
    on = 2 * NI + 2 * R
    ov = on + R
    o0 = ov + R
    ors = o0 + R
    onr = ors + R
    # rstd = sqrt(K) * r0 * (1.5 - 0.5 * vk * r0^2)   (one Newton step)
    tn = nl.ndarray((128, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tn, data1=st[:, o0:o0 + R], data2=st[:, o0:o0 + R], op=nl.multiply)
    nisa.tensor_tensor(dst=tn, data1=tn, data2=st[:, ov:ov + R], op=nl.multiply)
    nisa.tensor_scalar(dst=tn, data=tn, op0=nl.multiply, operand0=-0.5 * sk,
                       op1=nl.add, operand1=1.5 * sk)
    nisa.tensor_tensor(dst=st[:, ors:ors + R], data1=st[:, o0:o0 + R], data2=tn, op=nl.multiply)
    if need_nmr == 1:
        nisa.tensor_tensor(dst=st[:, onr:onr + R], data1=st[:, on:on + R],
                           data2=st[:, ors:ors + R], op=nl.multiply)


def _apply(xs, st, wt, bt, t0, t1, t2, y0, y1, y2, y3, orr, p0, base, i0, i1, ctr0,
           R, NC, nblk, K, stt_mode):
    NI = R * NC
    on = 2 * NI + 2 * R
    ors = on + 3 * R
    onr = ors + R
    for i in range(i0, i1):
        j = i // NC
        c = i % NC
        a0 = _chunk_lo(c, NC, nblk, K)
        a1 = _chunk_lo(c + 1, NC, nblk, K)
        cw = a1 - a0
        o = j * K + a0
        q = ctr0 + i
        tt = _pick3(q % 3, t0, t1, t2)
        yt = _pick4(q % 4, y0, y1, y2, y3)
        if stt_mode == 2:
            nisa.scalar_tensor_tensor(dst=tt[:, 0:cw], data=xs[:, o:o + cw], op0=nl.add,
                                      operand0=st[:, on + j:on + j + 1], op1=nl.multiply,
                                      operand1=wt[:, a0:a1], engine=nisa.engine.vector)
            nisa.scalar_tensor_tensor(dst=yt[:, 0:cw], data=tt[:, 0:cw], op0=nl.multiply,
                                      operand0=st[:, ors + j:ors + j + 1], op1=nl.add,
                                      operand1=bt[:, a0:a1], engine=nisa.engine.vector)
        elif stt_mode == 1:
            nisa.scalar_tensor_tensor(dst=tt[:, 0:cw], data=xs[:, o:o + cw], op0=nl.add,
                                      operand0=st[:, on + j:on + j + 1], op1=nl.multiply,
                                      operand1=wt[:, a0:a1])
            nisa.scalar_tensor_tensor(dst=yt[:, 0:cw], data=tt[:, 0:cw], op0=nl.multiply,
                                      operand0=st[:, ors + j:ors + j + 1], op1=nl.add,
                                      operand1=bt[:, a0:a1])
        else:
            nisa.activation(dst=tt[:, 0:cw], op=nl.copy, data=xs[:, o:o + cw],
                            scale=st[:, ors + j:ors + j + 1], bias=st[:, onr + j:onr + j + 1])
            nisa.tensor_tensor(dst=tt[:, 0:cw], data1=tt[:, 0:cw], data2=wt[:, a0:a1],
                               op=nl.multiply, engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=yt[:, 0:cw], data1=tt[:, 0:cw], data2=bt[:, a0:a1],
                               op=nl.add, engine=nisa.engine.vector)
        nisa.dma_copy(dst=orr[p0:p0 + 128, base + o:base + o + cw], src=yt[:, 0:cw])


@nki.jit
def _ln_kernel(x, w, b, n_rows, k_cols, rows_per_group, n_chunks, chunk_cols, stt_mode, keps, sk):
    P = 128
    K = k_cols
    R = rows_per_group
    NC = n_chunks
    CW = chunk_cols
    NI = R * NC
    SW = 2 * NI + 7 * R

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rpp = n_rows // (n_prog * P)
    G = rpp // R
    F = R * K
    nblk = (K + 511) // 512
    p0 = pid * P
    xr = x.reshape((n_prog * P, rpp * K))
    orr = out.reshape((n_prog * P, rpp * K))
    w2 = w.reshape((1, K))
    b2 = b.reshape((1, K))
    need_nmr = 0
    if stt_mode == 0:
        need_nmr = 1

    xb0 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb1 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb2 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb3 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    wt = nl.ndarray((P, K), dtype=w.dtype, buffer=nl.sbuf)
    bt = nl.ndarray((P, K), dtype=b.dtype, buffer=nl.sbuf)
    junk = nl.ndarray((P, CW), dtype=nl.bfloat16, buffer=nl.sbuf)
    mhalf = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mhalf, value=-0.5)
    st0 = nl.ndarray((P, SW), dtype=nl.float32, buffer=nl.sbuf)
    st1 = nl.ndarray((P, SW), dtype=nl.float32, buffer=nl.sbuf)
    st2 = nl.ndarray((P, SW), dtype=nl.float32, buffer=nl.sbuf)
    st3 = nl.ndarray((P, SW), dtype=nl.float32, buffer=nl.sbuf)
    tb0 = nl.ndarray((P, CW), dtype=nl.float32, buffer=nl.sbuf)
    tb1 = nl.ndarray((P, CW), dtype=nl.float32, buffer=nl.sbuf)
    tb2 = nl.ndarray((P, CW), dtype=nl.float32, buffer=nl.sbuf)
    yb0 = nl.ndarray((P, CW), dtype=x.dtype, buffer=nl.sbuf)
    yb1 = nl.ndarray((P, CW), dtype=x.dtype, buffer=nl.sbuf)
    yb2 = nl.ndarray((P, CW), dtype=x.dtype, buffer=nl.sbuf)
    yb3 = nl.ndarray((P, CW), dtype=x.dtype, buffer=nl.sbuf)

    npre = min(3, G)
    for g in range(npre):
        _ld_group(_pick4(g, xb0, xb1, xb2, xb3), xr, p0, g * F, R, NC, nblk, K)
        if g == 0:
            for c in range(NC):
                a0 = _chunk_lo(c, NC, nblk, K)
                a1 = _chunk_lo(c + 1, NC, nblk, K)
                nisa.dma_copy(dst=wt[:, a0:a1], src=w2.ap(pattern=[[0, P], [1, a1 - a0]], offset=a0))
                nisa.dma_copy(dst=bt[:, a0:a1], src=b2.ap(pattern=[[0, P], [1, a1 - a0]], offset=a0))

    # group 0: VectorE row sums in parallel with ScalarE sums of squares
    _st_group(xb0, st0, junk, R, NC, nblk, K, 1)
    if G > 1:
        _st_group(xb1, st1, junk, R, NC, nblk, K, 0)
    _fin_a(st0, mhalf, R, NC, K, keps)
    _fin_b(st0, R, NC, sk, need_nmr)

    h = NI // 2
    for g in range(G):
        if g + 2 < G:
            _st_group(_pick4((g + 2) % 4, xb0, xb1, xb2, xb3),
                      _pick4((g + 2) % 4, st0, st1, st2, st3), junk, R, NC, nblk, K, 0)
        if g + 3 < G:
            _ld_group(_pick4((g + 3) % 4, xb0, xb1, xb2, xb3), xr, p0, (g + 3) * F, R, NC, nblk, K)
        xs = _pick4(g % 4, xb0, xb1, xb2, xb3)
        stc = _pick4(g % 4, st0, st1, st2, st3)
        stn = _pick4((g + 1) % 4, st0, st1, st2, st3)
        _apply(xs, stc, wt, bt, tb0, tb1, tb2, yb0, yb1, yb2, yb3, orr, p0, g * F, 0, h,
               g * NI, R, NC, nblk, K, stt_mode)
        if g + 1 < G:
            _fin_a(stn, mhalf, R, NC, K, keps)
        _apply(xs, stc, wt, bt, tb0, tb1, tb2, yb0, yb1, yb2, yb3, orr, p0, g * F, h, NI,
               g * NI, R, NC, nblk, K, stt_mode)
        if g + 1 < G:
            _fin_b(stn, R, NC, sk, need_nmr)

    return out


# ---------------- round-1 kernel (compile-failure fallback only) ----------------
@nki.jit
def _ln_kernel_r1(x, w, b, n_rows, k_cols, eps, rows_per_group, n_chunks, chunk_cols):
    P = 128
    K = k_cols
    R = rows_per_group
    NC = n_chunks
    CWB = chunk_cols

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rpp = n_rows // (n_prog * P)
    G = rpp // R
    F = R * K
    nblk = (K + 511) // 512
    inv_k = 1.0 / K
    neg_inv_k = -1.0 / K
    p0 = pid * P

    xr = x.reshape((n_prog * P, rpp * K))
    orr = out.reshape((n_prog * P, rpp * K))

    w2 = w.reshape((1, K))
    b2 = b.reshape((1, K))
    wt = nl.ndarray((P, F), dtype=w.dtype, buffer=nl.sbuf)
    bt = nl.ndarray((P, F), dtype=b.dtype, buffer=nl.sbuf)
    for j in range(R):
        nisa.dma_copy(dst=wt[:, j * K:(j + 1) * K], src=w2.ap(pattern=[[0, P], [1, K]], offset=0))
        nisa.dma_copy(dst=bt[:, j * K:(j + 1) * K], src=b2.ap(pattern=[[0, P], [1, K]], offset=0))

    mhalf = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mhalf, value=-0.5)
    junk = nl.ndarray((P, CWB), dtype=nl.bfloat16, buffer=nl.sbuf)

    xb0 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb1 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb2 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    rb0 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    rb1 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    rb2 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nb0 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nb1 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nb2 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    fb0 = nl.ndarray((P, CWB), dtype=nl.float32, buffer=nl.sbuf)
    fb1 = nl.ndarray((P, CWB), dtype=nl.float32, buffer=nl.sbuf)
    fb2 = nl.ndarray((P, CWB), dtype=nl.float32, buffer=nl.sbuf)
    yb0 = nl.ndarray((P, CWB), dtype=x.dtype, buffer=nl.sbuf)
    yb1 = nl.ndarray((P, CWB), dtype=x.dtype, buffer=nl.sbuf)
    yb2 = nl.ndarray((P, CWB), dtype=x.dtype, buffer=nl.sbuf)

    for s in range(G + 2):
        s1 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        s2 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        s1p = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
        s2p = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)

        if s < G:
            xa = _pick3(s % 3, xb0, xb1, xb2)
            ca = s * F
            if NC > 1:
                for c in range(NC):
                    a0 = _chunk_lo(c, NC, nblk, K)
                    a1 = _chunk_lo(c + 1, NC, nblk, K)
                    nisa.dma_copy(dst=xa[:, a0:a1], src=xr[p0:p0 + P, ca + a0:ca + a1])
            else:
                nisa.dma_copy(dst=xa[:, 0:F], src=xr[p0:p0 + P, ca:ca + F])

        if s >= 1:
            if s <= G:
                gb = s - 1
                xs = _pick3(gb % 3, xb0, xb1, xb2)
                if NC > 1:
                    for c in range(NC):
                        a0 = _chunk_lo(c, NC, nblk, K)
                        a1 = _chunk_lo(c + 1, NC, nblk, K)
                        nisa.tensor_reduce(dst=s1p[:, c:c + 1], op=nl.add, data=xs[:, a0:a1], axis=1)
                        nisa.activation(dst=junk[:, 0:a1 - a0], op=nl.square, data=xs[:, a0:a1],
                                        reduce_op=nl.add, reduce_res=s2p[:, c:c + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                else:
                    for j in range(R):
                        nisa.tensor_reduce(dst=s1[:, j:j + 1], op=nl.add,
                                           data=xs[:, j * K:(j + 1) * K], axis=1)
                        nisa.activation(dst=junk[:, 0:K], op=nl.square, data=xs[:, j * K:(j + 1) * K],
                                        reduce_op=nl.add, reduce_res=s2[:, j:j + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)

        if s >= 2:
            gc = s - 2
            xc = _pick3(gc % 3, xb0, xb1, xb2)
            rsc = _pick3(gc % 3, rb0, rb1, rb2)
            nmc = _pick3(gc % 3, nb0, nb1, nb2)
            cc0 = gc * F
            if R == 1:
                for c in range(NC):
                    a0 = _chunk_lo(c, NC, nblk, K)
                    a1 = _chunk_lo(c + 1, NC, nblk, K)
                    cw = a1 - a0
                    q = (gc * NC + c) % 3
                    ft = _pick3(q, fb0, fb1, fb2)
                    yt = _pick3(q, yb0, yb1, yb2)
                    nisa.activation(dst=ft[:, 0:cw], op=nl.copy, data=xc[:, a0:a1],
                                    scale=rsc, bias=nmc)
                    nisa.tensor_tensor(dst=ft[:, 0:cw], data1=ft[:, 0:cw], data2=wt[:, a0:a1],
                                       op=nl.multiply, engine=nisa.engine.vector)
                    nisa.tensor_tensor(dst=yt[:, 0:cw], data1=ft[:, 0:cw], data2=bt[:, a0:a1],
                                       op=nl.add, engine=nisa.engine.gpsimd)
                    nisa.dma_copy(dst=orr[p0:p0 + P, cc0 + a0:cc0 + a1], src=yt[:, 0:cw])
            else:
                q = gc % 3
                ft = _pick3(q, fb0, fb1, fb2)
                yt = _pick3(q, yb0, yb1, yb2)
                for j in range(R):
                    nisa.activation(dst=ft[:, j * K:(j + 1) * K], op=nl.copy,
                                    data=xc[:, j * K:(j + 1) * K],
                                    scale=rsc[:, j:j + 1], bias=nmc[:, j:j + 1])
                nisa.tensor_tensor(dst=ft[:, 0:F], data1=ft[:, 0:F], data2=wt[:, 0:F],
                                   op=nl.multiply, engine=nisa.engine.vector)
                nisa.tensor_tensor(dst=yt[:, 0:F], data1=ft[:, 0:F], data2=bt[:, 0:F],
                                   op=nl.add, engine=nisa.engine.gpsimd)
                nisa.dma_copy(dst=orr[p0:p0 + P, cc0:cc0 + F], src=yt[:, 0:F])

        if s >= 1:
            if s <= G:
                gb = s - 1
                rs = _pick3(gb % 3, rb0, rb1, rb2)
                nm = _pick3(gb % 3, nb0, nb1, nb2)
                if NC > 1:
                    nisa.tensor_reduce(dst=s1, op=nl.add, data=s1p, axis=1)
                    nisa.tensor_reduce(dst=s2, op=nl.add, data=s2p, axis=1)
                negmean = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=negmean, data=s1, op0=nl.multiply, operand0=neg_inv_k)
                msq = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=msq, data1=negmean, data2=negmean, op=nl.multiply)
                vr = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=vr, data=s2, op0=nl.multiply, operand0=inv_k)
                nisa.tensor_tensor(dst=vr, data1=vr, data2=msq, op=nl.subtract)
                nisa.tensor_scalar(dst=vr, data=vr, op0=nl.maximum, operand0=0.0,
                                   op1=nl.add, operand1=eps)
                r0 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=r0, data1=vr, data2=mhalf, op=nl.power,
                                   engine=nisa.engine.gpsimd)
                tn = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tn, data1=r0, data2=r0, op=nl.multiply)
                nisa.tensor_tensor(dst=tn, data1=tn, data2=vr, op=nl.multiply)
                nisa.tensor_scalar(dst=tn, data=tn, op0=nl.multiply, operand0=-0.5,
                                   op1=nl.add, operand1=1.5)
                nisa.tensor_tensor(dst=rs, data1=r0, data2=tn, op=nl.multiply)
                nisa.tensor_tensor(dst=nm, data1=negmean, data2=rs, op=nl.multiply)

    return out


def _make_config(n_rows, k):
    rpp = n_rows // (2 * _P)
    nblk = (k + 511) // 512
    R = 1
    if 4 * k <= 3072 and rpp % 4 == 0 and rpp >= 8:
        R = 4
    elif 2 * k <= 4096 and rpp % 2 == 0 and rpp >= 4:
        R = 2
    if R > 1:
        NC = 1
    else:
        NC = (nblk + 3) // 4
        if NC < 2 and nblk >= 2:
            NC = 2
    cw = ((nblk + NC - 1) // NC) * 512
    if cw > k:
        cw = k
    return {"rows_per_group": R, "n_chunks": NC, "chunk_cols": cw,
            "x_buffers": 4, "stats_ahead": 2, "fin_ahead": 1,
            "lnc_programs": 2, "partitions": _P}


def _cfg_r1(n_rows, k):
    rpp = n_rows // (2 * _P)
    nblk = (k + 511) // 512
    if k <= 1024 and rpp % 2 == 0:
        R = 2
        NC = 1
        cwb = 2 * k
    else:
        R = 1
        NC = (nblk + 3) // 4
        cwb = ((nblk + NC - 1) // NC) * 512
        if cwb > k:
            cwb = k
    return {"rows_per_group": R, "n_chunks": NC, "chunk_buf_cols": cwb,
            "lnc_programs": 2, "partitions": _P}


def _launch(mode, x, w, b, n_rows, K, eps, cfg, cfg1):
    if mode < 3:
        return _ln_kernel(x, w, b, n_rows, K, cfg["rows_per_group"], cfg["n_chunks"],
                          cfg["chunk_cols"], 2 - mode, float(K) * eps, math.sqrt(float(K)))
    return _ln_kernel_r1(x, w, b, n_rows, K, eps, cfg1["rows_per_group"], cfg1["n_chunks"],
                         cfg1["chunk_buf_cols"])


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    global _LAST_CONFIG
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    K = int(x.shape[-1])
    n_rows = int(x.numel() // K)
    assert n_rows % (2 * _P) == 0, "row count must be a multiple of 256"
    cfg = _make_config(n_rows, K)
    cfg1 = _cfg_r1(n_rows, K)
    key = (n_rows, K, str(x.dtype))
    mode = _PATH_CACHE.get(key, 0)
    out = None
    err = None
    while out is None and mode < 4:
        try:
            out = _launch(mode, x, weight, bias, n_rows, K, float(eps), cfg, cfg1)
        except Exception as e:  # compile-time API incompatibility -> next code path
            err = e
            out = None
            mode += 1
    if out is None:
        raise err
    _PATH_CACHE[key] = mode
    if mode < 3:
        c = dict(cfg)
    else:
        c = dict(cfg1)
    c["path"] = _PATH_NAMES[mode]
    _LAST_CONFIG = c
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
