# Half-split ("rotate-half") RoPE for q (B, S, H, D) with tables cos/sin (S, D/2).
#
# Single NKI kernel, out-of-place.
#   * Each of the two physical cores (LNC=2) owns a contiguous half of every batch's
#     sequence. Inside a core, partition p owns K = S/(2*128) consecutive rows; tile j
#     processes row j of every partition (one 8 KiB contiguous row per partition).
#   * Tables for a partition are therefore contiguous: they are loaded in chunks of
#     tc rows (large descriptors) through a small ring, and -sin is formed once per chunk.
#   * Products in the input dtype (bit-identical to the reference's fp16 products):
#       VectorE: q1*c, q2*(-s), q2*c        GpSimd (concurrently): q1*s
#   * TensorE adds the pairs with identity matmuls into fp32 PSUM (one rounding only),
#     ScalarE evacuates PSUM to fp16 into the natural [h][half][i] layout.
#   * Explicit ring buffers decouple DMA / VectorE / GpSimd / TensorE / ScalarE.
import nki
import nki.language as nl
import nki.isa as nisa

_MAX_ROWS = 128      # partitions per tile
_NSPLIT = 8          # DMA slabs per tile load / store
_PSUM_W = 1024       # PSUM slot width in fp32 columns (2 banks)
_MM_N = 512          # max matmul moving free size
_PREFETCH = 3        # tiles prefetched ahead (q ring depth = 4)
_TAB_CHUNK_MAX = 16  # max table rows per partition per chunk load

_LAST_CONFIG = {}


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


def _issue_loads(q5, c4, s4, qd, cd, sd, b, pid, j, sp, nsplit, tc, load_tab):
    if load_tab:
        nisa.dma_copy(dst=cd, src=c4[pid, :, j:j + tc, :])
        nisa.dma_copy(dst=sd, src=s4[pid, :, j:j + tc, :])
    for jj in range(nsplit):
        p0 = jj * sp
        nisa.dma_copy(dst=qd[p0:p0 + sp, :], src=q5[b, pid, p0:p0 + sp, j, :])


def _issue_stores(o5, od, b, pid, j, sp, nsplit):
    for jj in range(nsplit):
        p0 = jj * sp
        nisa.dma_copy(dst=o5[b, pid, p0:p0 + sp, j, :], src=od[p0:p0 + sp, :])


@nki.jit
def _rope_kernel(q, cos, sin, pr, nsplit, psw, mmn, pf, tc):
    B = q.shape[0]
    S = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]
    D2 = D // 2
    F = H * D
    F2 = H * D2
    dt = q.dtype

    out = nl.ndarray((B, S, H, D), dtype=dt, buffer=nl.shared_hbm)

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    K = S // (nprog * pr)
    q5 = q.reshape((B, nprog, pr, K, F))
    o5 = out.reshape((B, nprog, pr, K, F))
    c4 = cos.reshape((nprog, pr, K, D2))
    s4 = sin.reshape((nprog, pr, K, D2))
    ntl = B * K
    sp = pr // nsplit

    # ---- ring buffers (separate allocations, live for the whole loop) ----
    q_a = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    q_b = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    q_c = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    q_d = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    c_a = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    c_b = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    c_c = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    s_a = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    s_b = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    s_c = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    n_a = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    n_b = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    n_c = nl.ndarray((pr, tc, D2), dtype=dt, buffer=nl.sbuf)
    pa_a = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pa_b = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pa_c = nl.ndarray((pr, 2 * F2), dtype=dt, buffer=nl.sbuf)
    pb_a = nl.ndarray((pr, F2), dtype=dt, buffer=nl.sbuf)
    pb_b = nl.ndarray((pr, F2), dtype=dt, buffer=nl.sbuf)
    pb_c = nl.ndarray((pr, F2), dtype=dt, buffer=nl.sbuf)
    pg_a = nl.ndarray((pr, F2), dtype=dt, buffer=nl.sbuf)
    pg_b = nl.ndarray((pr, F2), dtype=dt, buffer=nl.sbuf)
    pg_c = nl.ndarray((pr, F2), dtype=dt, buffer=nl.sbuf)
    o_a = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    o_b = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    o_c = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    ps_a = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)
    ps_b = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)
    ps_c = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)

    # ---- prologue: start the first loads as early as possible ----
    npre = min(pf, ntl)
    for k in range(npre):
        bp = k // K
        jp = k - bp * K
        gp = k // tc
        ltp = False
        if k % tc == 0:
            ltp = True
        _issue_loads(q5, c4, s4,
                     _pick4(k % 4, q_a, q_b, q_c, q_d),
                     _pick3(gp % 3, c_a, c_b, c_c),
                     _pick3(gp % 3, s_a, s_b, s_c),
                     bp, pid, jp, sp, nsplit, tc, ltp)

    # ---- +I (input dtype) for the TensorE accumulation ----
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

    ngh = F2 // psw        # PSUM groups per half
    nh = psw // D2         # heads per PSUM group
    nmm = psw // mmn       # matmul column chunks per group
    gcount = 0

    for k in range(ntl):
        # prefetch tile k + pf into the slot freed by tile k - 1
        kn = k + pf
        if kn < ntl:
            bn = kn // K
            jn = kn - bn * K
            gn = kn // tc
            lt = False
            if kn % tc == 0:
                lt = True
            _issue_loads(q5, c4, s4,
                         _pick4(kn % 4, q_a, q_b, q_c, q_d),
                         _pick3(gn % 3, c_a, c_b, c_c),
                         _pick3(gn % 3, s_a, s_b, s_c),
                         bn, pid, jn, sp, nsplit, tc, lt)

        b = k // K
        j = k - b * K
        g = k // tc
        jl = k - g * tc
        qs = _pick4(k % 4, q_a, q_b, q_c, q_d)
        ct = _pick3(g % 3, c_a, c_b, c_c)
        st = _pick3(g % 3, s_a, s_b, s_c)
        nt = _pick3(g % 3, n_a, n_b, n_c)
        pa = _pick3(k % 3, pa_a, pa_b, pa_c)
        pb = _pick3(k % 3, pb_a, pb_b, pb_c)
        pg = _pick3(k % 3, pg_a, pg_b, pg_c)
        osl = _pick3(k % 3, o_a, o_b, o_c)

        if jl == 0:
            # -sin for this table chunk (exact)
            nisa.tensor_scalar(dst=nt, data=st, op0=nl.multiply, operand0=-1.0)

        q3 = qs.reshape((pr, H, D))
        cbr = ct[:, jl:jl + 1, :].broadcast(1, H)
        sbr = st[:, jl:jl + 1, :].broadcast(1, H)
        nbr = nt[:, jl:jl + 1, :].broadcast(1, H)
        pa3 = pa.reshape((pr, 2 * H, D2))
        pb3 = pb.reshape((pr, H, D2))
        pg3 = pg.reshape((pr, H, D2))

        # products in the input dtype
        nisa.tensor_tensor(dst=pa3[:, 0:H, :], data1=q3[:, :, 0:D2],
                           data2=cbr, op=nl.multiply, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=pa3[:, H:2 * H, :], data1=q3[:, :, D2:D],
                           data2=nbr, op=nl.multiply, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=pg3, data1=q3[:, :, 0:D2],
                           data2=sbr, op=nl.multiply, engine=nisa.engine.gpsimd)
        nisa.tensor_tensor(dst=pb3, data1=q3[:, :, D2:D],
                           data2=cbr, op=nl.multiply, engine=nisa.engine.vector)

        o3 = osl.reshape((pr, H, D))
        for half in range(2):
            if half == 0:
                srca = pa
                offa = 0
                srcb = pa
                offb = F2
            else:
                srca = pb
                offa = 0
                srcb = pg
                offb = 0
            for gi in range(ngh):
                ps = _pick3(gcount % 3, ps_a, ps_b, ps_c)
                gcount += 1
                off = gi * psw
                for jm in range(nmm):
                    j0 = jm * mmn
                    a0 = offa + off + j0
                    b0 = offb + off + j0
                    nisa.nc_matmul(dst=ps[:, j0:j0 + mmn], stationary=ident,
                                   moving=srca[:, a0:a0 + mmn],
                                   accumulate=False)
                    nisa.nc_matmul(dst=ps[:, j0:j0 + mmn], stationary=ident,
                                   moving=srcb[:, b0:b0 + mmn],
                                   accumulate=True)
                h0 = off // D2
                nisa.activation(dst=o3[:, h0:h0 + nh, half * D2:(half + 1) * D2],
                                op=nl.copy, data=ps.reshape((pr, nh, D2)))

        _issue_stores(o5, osl, b, pid, j, sp, nsplit)

    return out


def _rows_per_partition_tile(S):
    pr = _MAX_ROWS
    while pr > 1 and S % (2 * pr) != 0:
        pr //= 2
    return pr


def _table_chunk(K):
    tc = 1
    for cand in range(min(_TAB_CHUNK_MAX, K), 0, -1):
        if K % cand == 0:
            tc = cand
            break
    return tc


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
    tc = _table_chunk(K)

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
        "table_chunk_rows": tc,
        "dma_slabs_per_tile": nsplit,
        "prefetch_tiles": _PREFETCH,
        "q_ring": 4,
        "table_ring": 3,
        "product_ring": 3,
        "output_ring": 3,
        "psum_ring": 3,
        "psum_width": psw,
        "heads_per_psum_group": hpg,
        "matmul_n": mmn,
        "lnc_split": "contiguous half of sequence per program, contiguous rows per partition",
        "products": "vector: q1*c, q2*(-s), q2*c; gpsimd: q1*s (input dtype)",
        "combine": "tensor engine identity matmuls in fp32 psum, scalar engine evacuation",
    }
    out = _rope_kernel(q, cos, sin, pr, nsplit, psw, mmn, _PREFETCH, tc)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
