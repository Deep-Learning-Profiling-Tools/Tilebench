# Half-split ("rotate-half") RoPE for q (B, S, H, D) with tables cos/sin (S, D/2).
#
# Single NKI kernel, out-of-place.
#   * LNC=2: each physical core owns a contiguous half of every batch's sequence.
#     Partition p owns K = S/(2*128) consecutive rows; a tile covers rr consecutive
#     rows of every partition -> one contiguous rr*8 KiB chunk per partition per DMA
#     (rr = 2 for larger problems: bigger DMA descriptors, half the DMA count per byte).
#   * cos/sin of the core's rows stay resident in SBUF (small first chunk for a fast
#     start, then the rest in one large contiguous-per-partition DMA).
#   * VectorE forms the four products in the input dtype (bit-identical to the
#     reference products):  pa = [q1*c | q2*s],  pb = [q2*c | q1*s]
#   * TensorE adds them with +I / -I identity matmuls into fp32 PSUM (one rounding),
#     ScalarE evacuates PSUM -> fp16 into the natural [h][half][i] layout.
#   * Separately allocated ring buffers decouple DMA / VectorE / TensorE / ScalarE.
import nki
import nki.language as nl
import nki.isa as nisa

_MAX_ROWS = 128      # partitions per tile
_NSPLIT = 8          # DMA slabs per tile load / store
_PSUM_W = 1024       # PSUM slot width in fp32 columns (2 banks)
_MM_N = 512          # max matmul moving free size
_PREFETCH = 2        # tiles prefetched ahead (q ring depth = 3)
_TAB_FIRST = 4       # table rows per partition in the first (fast-start) chunk
_RR_MIN_K = 12       # use 2 rows per partition per tile when K >= this

_LAST_CONFIG = {}


def _sel3(i, a, b, c):
    r = c
    if i == 0:
        r = a
    elif i == 1:
        r = b
    return r


def _load_q(q5, qd, b, pid, kt, sp, nsplit):
    for jj in range(nsplit):
        p0 = jj * sp
        nisa.dma_copy(dst=qd[p0:p0 + sp, :], src=q5[b, pid, p0:p0 + sp, kt, :])


def _store_o(o5, od, b, pid, kt, sp, nsplit):
    for jj in range(nsplit):
        p0 = jj * sp
        nisa.dma_copy(dst=o5[b, pid, p0:p0 + sp, kt, :], src=od[p0:p0 + sp, :])


@nki.jit
def _rope_kernel(q, cos, sin, pr, rr, nsplit, psw, mmn, pf, c0):
    B = q.shape[0]
    S = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]
    D2 = D // 2
    F = H * D
    F2 = H * D2
    FT = rr * F
    dt = q.dtype

    out = nl.ndarray((B, S, H, D), dtype=dt, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    K = S // (nprog * pr)
    KT = K // rr
    q5 = q.reshape((B, nprog, pr, KT, FT))
    o5 = out.reshape((B, nprog, pr, KT, FT))
    c4 = cos.reshape((nprog, pr, K, D2))
    s4 = sin.reshape((nprog, pr, K, D2))
    ntl = B * KT
    sp = pr // nsplit
    cf0 = min(c0, K)
    rest = K - cf0
    rs = max(1, rest)

    # ---- ring buffers (separate allocations, live for the whole loop) ----
    q_a = nl.ndarray((pr, FT), dtype=dt, buffer=nl.sbuf)
    q_b = nl.ndarray((pr, FT), dtype=dt, buffer=nl.sbuf)
    q_c = nl.ndarray((pr, FT), dtype=dt, buffer=nl.sbuf)
    o_a = nl.ndarray((pr, FT), dtype=dt, buffer=nl.sbuf)
    o_b = nl.ndarray((pr, FT), dtype=dt, buffer=nl.sbuf)
    o_c = nl.ndarray((pr, FT), dtype=dt, buffer=nl.sbuf)
    pa_a = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pa_b = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pa_c = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pb_a = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pb_b = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pb_c = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    ps_a = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)
    ps_b = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)
    ps_c = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)

    # ---- resident tables: first chunk + rest (separate allocations) ----
    cf = nl.ndarray((pr, cf0, D2), dtype=dt, buffer=nl.sbuf)
    sf = nl.ndarray((pr, cf0, D2), dtype=dt, buffer=nl.sbuf)
    cr = nl.ndarray((pr, rs, D2), dtype=dt, buffer=nl.sbuf)
    sr = nl.ndarray((pr, rs, D2), dtype=dt, buffer=nl.sbuf)

    # ---- prologue: start the first loads as early as possible ----
    nisa.dma_copy(dst=cf, src=c4[pid, :, 0:cf0, :])
    nisa.dma_copy(dst=sf, src=s4[pid, :, 0:cf0, :])
    npre = min(pf, ntl)
    for k in range(npre):
        bp = k // KT
        ktp = k - bp * KT
        _load_q(q5, _sel3(k % 3, q_a, q_b, q_c), bp, pid, ktp, sp, nsplit)
    if rest > 0:
        nisa.dma_copy(dst=cr, src=c4[pid, :, cf0:K, :])
        nisa.dma_copy(dst=sr, src=s4[pid, :, cf0:K, :])

    # ---- +I and -I (input dtype) for the TensorE accumulation ----
    iota_c = nl.ndarray((pr, pr), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iota_c, pattern=[[1, pr]], offset=0, channel_multiplier=0)
    iota_p = nl.ndarray((pr, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iota_p, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    col_f = nl.ndarray((pr, pr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=col_f, src=iota_c)
    row_f = nl.ndarray((pr, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=row_f, src=iota_p)
    ident_f = nl.ndarray((pr, pr), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=ident_f, data=col_f, op0=nl.equal, operand0=row_f)
    ident = nl.ndarray((pr, pr), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ident, src=ident_f)
    nident = nl.ndarray((pr, pr), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nident, data=ident_f, op0=nl.multiply, operand0=-1.0)

    ngh = F2 // psw        # PSUM groups per half
    nh = psw // D2         # heads per PSUM group
    nmm = psw // mmn       # matmul column chunks per group
    gcount = 0

    for k in range(ntl):
        # prefetch tile k + pf into the q slot freed by tile k - 1
        kn = k + pf
        if kn < ntl:
            bn = kn // KT
            ktn = kn - bn * KT
            _load_q(q5, _sel3(kn % 3, q_a, q_b, q_c), bn, pid, ktn, sp, nsplit)

        b = k // KT
        kt = k - b * KT
        qs = _sel3(k % 3, q_a, q_b, q_c)
        osl = _sel3(k % 3, o_a, o_b, o_c)
        q3 = qs.reshape((pr, rr * H, D))
        o3 = osl.reshape((pr, rr * H, D))

        for r in range(rr):
            jt = kt * rr + r
            m = k * rr + r
            if jt < cf0:
                cbr = cf[:, jt:jt + 1, :].broadcast(1, H)
                sbr = sf[:, jt:jt + 1, :].broadcast(1, H)
            else:
                jr = jt - cf0
                cbr = cr[:, jr:jr + 1, :].broadcast(1, H)
                sbr = sr[:, jr:jr + 1, :].broadcast(1, H)
            pa = _sel3(m % 3, pa_a, pa_b, pa_c)
            pb = _sel3(m % 3, pb_a, pb_b, pb_c)
            pa3 = pa.reshape((pr, 2 * H, D2))
            pb3 = pb.reshape((pr, 2 * H, D2))
            h_lo = r * H
            h_hi = h_lo + H

            # products in the input dtype (VectorE); half-0 operands first
            nisa.tensor_tensor(dst=pa3[:, 0:H, :], data1=q3[:, h_lo:h_hi, 0:D2],
                               data2=cbr, op=nl.multiply, engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=pa3[:, H:2 * H, :], data1=q3[:, h_lo:h_hi, D2:D],
                               data2=sbr, op=nl.multiply, engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=pb3[:, 0:H, :], data1=q3[:, h_lo:h_hi, D2:D],
                               data2=cbr, op=nl.multiply, engine=nisa.engine.vector)
            nisa.tensor_tensor(dst=pb3[:, H:2 * H, :], data1=q3[:, h_lo:h_hi, 0:D2],
                               data2=sbr, op=nl.multiply, engine=nisa.engine.vector)

            for half in range(2):
                if half == 0:
                    src = pa
                    st2 = nident
                else:
                    src = pb
                    st2 = ident
                for gi in range(ngh):
                    ps = _sel3(gcount % 3, ps_a, ps_b, ps_c)
                    gcount += 1
                    off = gi * psw
                    for jm in range(nmm):
                        j0 = jm * mmn
                        a0 = off + j0
                        nisa.nc_matmul(dst=ps[:, j0:j0 + mmn], stationary=ident,
                                       moving=src[:, a0:a0 + mmn],
                                       accumulate=False)
                    for jm in range(nmm):
                        j0 = jm * mmn
                        b0 = F2 + off + j0
                        nisa.nc_matmul(dst=ps[:, j0:j0 + mmn], stationary=st2,
                                       moving=src[:, b0:b0 + mmn],
                                       accumulate=True)
                    h0 = h_lo + off // D2
                    nisa.activation(dst=o3[:, h0:h0 + nh, half * D2:(half + 1) * D2],
                                    op=nl.copy, data=ps.reshape((pr, nh, D2)))

        _store_o(o5, osl, b, pid, kt, sp, nsplit)

    return out


def _rows_per_partition_tile(S):
    pr = _MAX_ROWS
    while pr > 1 and S % (2 * pr) != 0:
        pr //= 2
    return pr


def run(q, cos, sin):
    global _LAST_CONFIG
    B = int(q.shape[0])
    S = int(q.shape[1])
    H = int(q.shape[2])
    D = int(q.shape[3])
    D2 = D // 2
    pr = _rows_per_partition_tile(S)
    assert S % (2 * pr) == 0, "seq_len must be even"
    nsplit = min(_NSPLIT, pr)
    K = S // (2 * pr)          # rows per partition per core (LNC=2)
    rr = 1
    if K % 2 == 0 and K >= _RR_MIN_K:
        rr = 2
    c0 = min(K, _TAB_FIRST)

    hpg = 1
    for cand in range(H, 0, -1):
        if H % cand == 0 and cand * D2 <= _PSUM_W:
            hpg = cand
            break
    psw = hpg * D2
    mmn = 1
    for cand in range(min(_MM_N, psw), 0, -1):
        if psw % cand == 0:
            mmn = cand
            break

    _LAST_CONFIG = {
        "partitions_per_tile": pr,
        "rows_per_partition_per_core": K,
        "rows_per_partition_per_tile": rr,
        "table_first_chunk_rows": c0,
        "tables": "resident in SBUF (first chunk + rest)",
        "dma_slabs_per_tile": nsplit,
        "prefetch_tiles": _PREFETCH,
        "q_ring": 3,
        "product_ring": 3,
        "output_ring": 3,
        "psum_ring": 3,
        "psum_width": psw,
        "heads_per_psum_group": hpg,
        "matmul_n": mmn,
        "lnc_split": "contiguous half of sequence per program, contiguous rows per partition",
        "products": "vector engine only, input dtype: [q1*c|q2*s], [q2*c|q1*s]",
        "combine": "tensor engine +I/-I identity matmuls in fp32 psum, scalar engine evacuation",
    }
    out = _rope_kernel(q, cos, sin, pr, rr, nsplit, psw, mmn, _PREFETCH, c0)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
