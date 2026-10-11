# Half-split ("rotate-half") RoPE for q (B, S, H, D) with tables cos/sin (S, D/2).
#
# Single NKI kernel, out-of-place:
#   * rows r = b*S + s on the partition axis (128 per tile); free axis = all heads.
#     rows_per_tile divides S, so a tile lies in one batch and table row = r % S.
#   * Tiles are split between the two physical cores (LNC=2).
#   * VectorE forms the four products in the input dtype, stored half-major:
#       blk0 = q1*c, blk1 = q2*s, blk2 = q2*c, blk3 = q1*s     (each [P, H*D/2])
#     (tables broadcast over heads by a stride-0 view).
#   * TensorE adds them with identity matmuls into fp32 PSUM (exact for fp16):
#       out_first  = I*blk0 + (-I)*blk1,   out_second = I*blk2 + I*blk3
#     ScalarE evacuates PSUM to fp16 (RNE) and writes the natural [h][half][i] layout.
#   * Explicit ring buffers (separately allocated) give deep prefetch and decouple
#     DMA / VectorE / TensorE / ScalarE. Tile loads/stores are split into 8 contiguous
#     partition slabs so many DMAs are in flight.
import nki
import nki.language as nl
import nki.isa as nisa

_MAX_ROWS = 128      # partition rows per tile
_NSPLIT = 8          # DMA slabs per tile load / store
_PSUM_W = 1024       # PSUM slot width in fp32 columns (2 banks)
_MM_N = 512          # max matmul moving free size
_PREFETCH = 3        # tiles prefetched ahead (input ring depth = 4)

_LAST_CONFIG = {}


def _pick2(i, a, b):
    r = b
    if i == 0:
        r = a
    return r


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


def _issue_loads(q2, cos, sin, qd, cd, sd, r0, s0, pr, sp, nsplit):
    nisa.dma_copy(dst=cd, src=cos[s0:s0 + pr, :])
    nisa.dma_copy(dst=sd, src=sin[s0:s0 + pr, :])
    for j in range(nsplit):
        p0 = j * sp
        nisa.dma_copy(dst=qd[p0:p0 + sp, :], src=q2[r0 + p0:r0 + p0 + sp, :])


def _issue_stores(o2, od, r0, sp, nsplit):
    for j in range(nsplit):
        p0 = j * sp
        nisa.dma_copy(dst=o2[r0 + p0:r0 + p0 + sp, :], src=od[p0:p0 + sp, :])


@nki.jit
def _rope_kernel(q, cos, sin, pr, nsplit, psw, mmn, pf):
    B = q.shape[0]
    S = q.shape[1]
    H = q.shape[2]
    D = q.shape[3]
    D2 = D // 2
    R = B * S
    F = H * D
    F2 = H * D2
    dt = q.dtype

    out = nl.ndarray((B, S, H, D), dtype=dt, buffer=nl.shared_hbm)
    q2 = q.reshape((R, F))
    o2 = out.reshape((R, F))

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    T = R // pr
    per = (T + nprog - 1) // nprog
    t_begin = min(T, pid * per)
    t_end = min(T, t_begin + per)
    ntl = t_end - t_begin
    sp = pr // nsplit

    # ---- ring buffers (separate allocations, live for the whole loop) ----
    q_a = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    q_b = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    q_c = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    q_d = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    c_a = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    c_b = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    c_c = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    c_d = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    s_a = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    s_b = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    s_c = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    s_d = nl.ndarray((pr, D2), dtype=dt, buffer=nl.sbuf)
    p_a = nl.ndarray((pr, 4 * F2), dtype=dt, buffer=nl.sbuf)
    p_b = nl.ndarray((pr, 4 * F2), dtype=dt, buffer=nl.sbuf)
    o_a = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    o_b = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    o_c = nl.ndarray((pr, F), dtype=dt, buffer=nl.sbuf)
    ps_a = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)
    ps_b = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)
    ps_c = nl.ndarray((pr, psw), dtype=nl.float32, buffer=nl.psum)

    # ---- prologue: start the first loads as early as possible ----
    npre = min(pf, ntl)
    for k in range(npre):
        r0p = (t_begin + k) * pr
        slp = k % 4
        _issue_loads(q2, cos, sin,
                     _pick4(slp, q_a, q_b, q_c, q_d),
                     _pick4(slp, c_a, c_b, c_c, c_d),
                     _pick4(slp, s_a, s_b, s_c, s_d),
                     r0p, r0p % S, pr, sp, nsplit)

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
        # prefetch tile k + pf into the slot freed by tile k - 1
        kn = k + pf
        if kn < ntl:
            rn = (t_begin + kn) * pr
            sln = kn % 4
            _issue_loads(q2, cos, sin,
                         _pick4(sln, q_a, q_b, q_c, q_d),
                         _pick4(sln, c_a, c_b, c_c, c_d),
                         _pick4(sln, s_a, s_b, s_c, s_d),
                         rn, rn % S, pr, sp, nsplit)

        r0 = (t_begin + k) * pr
        sl = k % 4
        qs = _pick4(sl, q_a, q_b, q_c, q_d)
        cs = _pick4(sl, c_a, c_b, c_c, c_d)
        ss = _pick4(sl, s_a, s_b, s_c, s_d)
        prod = _pick2(k % 2, p_a, p_b)
        osl = _pick3(k % 3, o_a, o_b, o_c)

        q3 = qs.reshape((pr, H, D))
        cbr = cs.reshape((pr, 1, D2)).broadcast(1, H)
        sbr = ss.reshape((pr, 1, D2)).broadcast(1, H)
        p3 = prod.reshape((pr, 4 * H, D2))

        # products (input dtype), half-major blocks
        nisa.tensor_tensor(dst=p3[:, 0:H, :], data1=q3[:, :, 0:D2],
                           data2=cbr, op=nl.multiply)
        nisa.tensor_tensor(dst=p3[:, H:2 * H, :], data1=q3[:, :, D2:D],
                           data2=sbr, op=nl.multiply)
        nisa.tensor_tensor(dst=p3[:, 2 * H:3 * H, :], data1=q3[:, :, D2:D],
                           data2=cbr, op=nl.multiply)
        nisa.tensor_tensor(dst=p3[:, 3 * H:4 * H, :], data1=q3[:, :, 0:D2],
                           data2=sbr, op=nl.multiply)

        o3 = osl.reshape((pr, H, D))
        for half in range(2):
            if half == 0:
                st2 = nident
            else:
                st2 = ident
            for gi in range(ngh):
                ps = _pick3(gcount % 3, ps_a, ps_b, ps_c)
                gcount += 1
                off = gi * psw
                ba = (2 * half) * F2 + off
                bb = (2 * half + 1) * F2 + off
                for j in range(nmm):
                    j0 = j * mmn
                    nisa.nc_matmul(dst=ps[:, j0:j0 + mmn], stationary=ident,
                                   moving=prod[:, ba + j0:ba + j0 + mmn],
                                   accumulate=False)
                    nisa.nc_matmul(dst=ps[:, j0:j0 + mmn], stationary=st2,
                                   moving=prod[:, bb + j0:bb + j0 + mmn],
                                   accumulate=True)
                h0 = off // D2
                nisa.activation(dst=o3[:, h0:h0 + nh, half * D2:(half + 1) * D2],
                                op=nl.copy, data=ps.reshape((pr, nh, D2)))

        _issue_stores(o2, osl, r0, sp, nsplit)

    return out


def _rows_per_tile(S):
    pr = _MAX_ROWS
    while pr > 1 and S % pr != 0:
        pr //= 2
    return pr


def run(q, cos, sin):
    global _LAST_CONFIG
    B = int(q.shape[0])
    S = int(q.shape[1])
    H = int(q.shape[2])
    D = int(q.shape[3])
    D2 = D // 2
    pr = _rows_per_tile(S)
    nsplit = min(_NSPLIT, pr)

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
        "rows_per_tile": pr,
        "dma_slabs_per_tile": nsplit,
        "prefetch_tiles": _PREFETCH,
        "input_ring": 4,
        "product_ring": 2,
        "output_ring": 3,
        "psum_ring": 3,
        "psum_width": psw,
        "heads_per_psum_group": hpg,
        "matmul_n": mmn,
        "lnc_split": "tiles split across 2 programs",
        "products": "vector engine, input dtype, half-major blocks",
        "combine": "tensor engine +I/-I identity matmuls in fp32 psum, scalar engine evacuation",
    }
    out = _rope_kernel(q, cos, sin, pr, nsplit, psw, mmn, _PREFETCH)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
