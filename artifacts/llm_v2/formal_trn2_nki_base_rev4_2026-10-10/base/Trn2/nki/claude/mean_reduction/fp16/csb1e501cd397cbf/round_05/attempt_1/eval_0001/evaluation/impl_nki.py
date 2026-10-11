import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (deterministic functions of the input shape only)
# ----------------------------------------------------------------------------
P = 128                         # partitions per tile
PREFETCH = 4                    # DMA units issued ahead of compute (program order)
MAIN_BYTES = 16384              # bytes / partition of a regular DMA unit (2 MiB per DMA)
LAST_MAIN_BYTES = 8192          # chunk size for the bulk of the final row
TAIL_BYTES = (1024, 2048, 4096) # final units of the final row (from the end)
ALIGN_BYTES = 1024              # chunk sizes are multiples of this (per partition)
MM_W = 512                      # identity-matmul moving width (one fp32 PSUM bank)
NO_T_TAIL = 3                   # last units without TensorE work (short tail)
GRAN = 64                       # V/S split granularity (elements)

_LAST_CONFIG = {}


@nki.jit
def _row_mean_kernel(x, u_off, u_len, u_g0, u_ng,
                     g_a, g_w, g_row, g_wt, g_wv, g_cv, g_cs, g_tf, g_tl, g_rl,
                     r_tcol, r_vc0, r_vns, r_sc0, r_sns,
                     n_units, nv_cols, ns_cols, junk_w, junk32_w, use_t, inv_n):
    # Program pid owns rows [pid*P*RP, (pid+1)*P*RP); partition p holds rows
    # pid*P*RP + p*RP + r (r in [0, RP)), contiguous in HBM.
    M = x.shape[0]
    N = x.shape[1]
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    RP = M // (n_prog * P)

    out = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    x2 = x.reshape((n_prog, P, RP * N))
    ov = out.reshape((n_prog, P, RP))

    # fp32 partial slots / row totals owned by VectorE and by ScalarE
    vparts = nl.ndarray((P, nv_cols), dtype=nl.float32, buffer=nl.sbuf)
    sparts = nl.ndarray((P, ns_cols), dtype=nl.float32, buffer=nl.sbuf)
    vt = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    st = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    # write-only scratch for ScalarE (only the fused fp32 sums are used)
    junk = nl.ndarray((P, junk_w), dtype=x.dtype, buffer=nl.sbuf)
    junk32 = nl.ndarray((P, junk32_w), dtype=nl.float32, buffer=nl.sbuf)

    ident = None
    b1 = None
    b2 = None
    b3 = None
    b4 = None
    acc = None
    n_mm = 0
    pt_on = False
    pt_acc = None
    pt_col = 0
    pv_on = False
    pv_row = 0

    for it in range(n_units + PREFETCH):
        tn = None
        if it < n_units:
            off = u_off[it]
            ln = u_len[it]
            tn = nl.ndarray((P, ln), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tn, src=x2[pid, :, off:off + ln])
        if it == PREFETCH - 1:
            if use_t:
                # identity matrix (built after the first DMAs are issued)
                iota_j = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(dst=iota_j, pattern=[[1, P]], offset=0, channel_multiplier=0)
                iota_p = nl.ndarray((P, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.iota(dst=iota_p, pattern=[[1, 1]], offset=0, channel_multiplier=1)
                f_j = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=f_j, src=iota_j)
                f_p = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=f_p, src=iota_p)
                ident_f = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ident_f, data=f_j, op0=nl.equal, operand0=f_p)
                ident = nl.ndarray((P, P), dtype=x.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=ident, src=ident_f)
        if it >= PREFETCH:
            u = it - PREFETCH
            t = b1
            # deferred VectorE work from the previous unit (independent of t)
            if pt_on:
                nisa.tensor_reduce(dst=vparts[:, pt_col:pt_col + 1], op=nl.add,
                                   data=pt_acc, axis=1)
                pt_on = False
            if pv_on:
                pvc0 = r_vc0[pv_row]
                pvns = r_vns[pv_row]
                nisa.tensor_reduce(dst=vt[:, pv_row:pv_row + 1], op=nl.add,
                                   data=vparts[:, pvc0:pvc0 + pvns], axis=1)
                pv_on = False
            g0 = u_g0[u]
            ng = u_ng[u]
            for gi in range(ng):
                g = g0 + gi
                a = g_a[g]
                w = g_w[g]
                r = g_row[g]
                wt = g_wt[g]
                wv = g_wv[g]
                ws = w - wt - wv
                # TensorE: identity matmul, fp32 PSUM accumulation per row
                if wt > 0:
                    if g_tf[g] > 0:
                        acc = nl.ndarray((P, MM_W), dtype=nl.float32, buffer=nl.psum)
                        n_mm = 0
                    for jj in range(wt // MM_W):
                        c = a + jj * MM_W
                        nisa.nc_matmul(dst=acc, stationary=ident,
                                       moving=t[:, c:c + MM_W],
                                       accumulate=(n_mm > 0))
                        n_mm = n_mm + 1
                # VectorE part
                v0 = a + wt
                cv = g_cv[g]
                if wv > 0:
                    if cv >= 0:
                        nisa.tensor_reduce(dst=vparts[:, cv:cv + 1], op=nl.add,
                                           data=t[:, v0:v0 + wv], axis=1)
                    else:
                        nisa.tensor_reduce(dst=vt[:, r:r + 1], op=nl.add,
                                           data=t[:, v0:v0 + wv], axis=1)
                # ScalarE part
                s0 = v0 + wv
                cs = g_cs[g]
                if ws > 0:
                    if cs >= 0:
                        nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                        data=t[:, s0:s0 + ws], reduce_op=nl.add,
                                        reduce_res=sparts[:, cs:cs + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    else:
                        nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                        data=t[:, s0:s0 + ws], reduce_op=nl.add,
                                        reduce_res=st[:, r:r + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                # ScalarE row total (rows other than the last one)
                if g_rl[g] > 0:
                    sns = r_sns[r]
                    if sns > 0:
                        if r < RP - 1:
                            sc0 = r_sc0[r]
                            nisa.activation(dst=junk32[:, 0:sns], op=nl.copy,
                                            data=sparts[:, sc0:sc0 + sns],
                                            reduce_op=nl.add,
                                            reduce_res=st[:, r:r + 1],
                                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
                if g_tl[g] > 0:
                    pt_on = True
                    pt_acc = acc
                    pt_col = r_tcol[r]
                if g_rl[g] > 0:
                    vns = r_vns[r]
                    if vns > 0:
                        if pt_on:
                            pv_on = True
                            pv_row = r
                        else:
                            vc0 = r_vc0[r]
                            nisa.tensor_reduce(dst=vt[:, r:r + 1], op=nl.add,
                                               data=vparts[:, vc0:vc0 + vns], axis=1)
        b1 = b2
        b2 = b3
        b3 = b4
        b4 = tn

    if pt_on:
        nisa.tensor_reduce(dst=vparts[:, pt_col:pt_col + 1], op=nl.add,
                           data=pt_acc, axis=1)
    if pv_on:
        pvc0 = r_vc0[pv_row]
        pvns = r_vns[pv_row]
        nisa.tensor_reduce(dst=vt[:, pv_row:pv_row + 1], op=nl.add,
                           data=vparts[:, pvc0:pvc0 + pvns], axis=1)
    # last row: VectorE folds the ScalarE partial slots (short final chain)
    lsns = r_sns[RP - 1]
    if lsns > 0:
        lsc0 = r_sc0[RP - 1]
        nisa.tensor_reduce(dst=st[:, RP - 1:RP], op=nl.add,
                           data=sparts[:, lsc0:lsc0 + lsns], axis=1)

    tot = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=vt, data2=st, op=nl.add)
    res = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res, data=tot, op0=nl.multiply, operand0=inv_n)
    nisa.dma_copy(dst=ov[pid], src=res)
    return out


def _even_chunks(n, main, align):
    if n <= 0:
        return []
    k = (n + main - 1) // main
    blocks = n // align
    if blocks < k:
        return [n]
    q, rr = divmod(blocks, k)
    res = []
    for i in range(k):
        res.append((q + (1 if i < rr else 0)) * align)
    res[0] += n - blocks * align
    return res


def _tail_chunks(n, main, align, tails):
    t = []
    rem = n
    for te in tails:
        if rem >= te:
            t.append(te)
            rem -= te
        else:
            break
    if 0 < rem < 2 * align and len(t) >= 2:
        t[-1] += rem
        rem = 0
    return _even_chunks(rem, main, align) + t[::-1]


def _plan(M, N, dtype):
    two_byte = dtype in (torch.float16, torch.bfloat16)
    isz = 2 if two_byte else 4
    if M % (2 * P) == 0:
        lnc = 2
    elif M % P == 0:
        lnc = 1
    else:
        raise ValueError("mean_reduction kernel requires M to be a multiple of 128")
    RP = M // (lnc * P)
    main = max(1, MAIN_BYTES // isz)
    main_last = max(1, LAST_MAIN_BYTES // isz)
    align = max(1, ALIGN_BYTES // isz)
    tails = [b // isz for b in TAIL_BYTES]
    rows_mode = (2 * N <= main)

    # ---- DMA units; each unit is a list of row segments (row, col0, width, local offset)
    units = []
    if rows_mode:
        kmax = 2
        while 2 * kmax * N <= main:
            kmax *= 2
        r = 0
        while r < RP - 1:
            k = kmax
            while k > RP - 1 - r:
                k = k // 2
            segs = []
            for j in range(k):
                segs.append((r + j, 0, N, j * N))
            units.append(segs)
            r += k
    else:
        for r in range(RP - 1):
            c0 = 0
            for w in _even_chunks(N, main, align):
                units.append([(r, c0, w, 0)])
                c0 += w
    c0 = 0
    for w in _tail_chunks(N, main_last, align, tails):
        units.append([(RP - 1, c0, w, 0)])
        c0 += w

    use_t = bool(two_byte and not rows_mode)
    nu = len(units)
    segs_all = []
    u_off, u_len, u_g0, u_ng = [], [], [], []
    for u in range(nu):
        su = units[u]
        u_off.append(int(su[0][0] * N + su[0][1]))
        tw = 0
        for s in su:
            tw += s[2]
        u_len.append(int(tw))
        u_g0.append(len(segs_all))
        u_ng.append(len(su))
        for s in su:
            r, c0, w, a = s
            wt = 0
            if use_t and len(su) == 1 and u < nu - NO_T_TAIL and w >= 2 * MM_W:
                wt = ((w // 2) // MM_W) * MM_W
            rem = w - wt
            if rem >= 2 * GRAN:
                wv = ((rem // 2) // GRAN) * GRAN
                wv = max(GRAN, min(rem - GRAN, wv))
            else:
                wv = max(1, rem // 2)
            segs_all.append((r, w, a, wt, wv))
    ng_tot = len(segs_all)

    # ---- per-row bookkeeping ----
    has_t = [False] * RP
    vseg = [0] * RP
    sseg = [0] * RP
    for (r, w, a, wt, wv) in segs_all:
        if wt > 0:
            has_t[r] = True
        if wv > 0:
            vseg[r] += 1
        if w - wt - wv > 0:
            sseg[r] += 1
    for r in range(RP):
        if sseg[r] == 0 or (vseg[r] == 0 and not has_t[r]):
            raise ValueError("unsupported split")
    v_direct = [(not has_t[r]) and vseg[r] == 1 for r in range(RP)]
    s_direct = [sseg[r] == 1 for r in range(RP)]

    vcnt = [0] * RP
    scnt = [0] * RP
    cv_slot = [-1] * ng_tot
    cs_slot = [-1] * ng_tot
    for g in range(ng_tot):
        r, w, a, wt, wv = segs_all[g]
        if wv > 0 and not v_direct[r]:
            cv_slot[g] = vcnt[r]
            vcnt[r] += 1
        if (w - wt - wv) > 0 and not s_direct[r]:
            cs_slot[g] = scnt[r]
            scnt[r] += 1
    tslot = [-1] * RP
    for r in range(RP):
        if has_t[r]:
            tslot[r] = vcnt[r]
            vcnt[r] += 1
    SMv = max(1, max(vcnt))
    SMs = max(1, max(scnt))

    first_t = {}
    last_t = {}
    last_g = {}
    for g in range(ng_tot):
        r = segs_all[g][0]
        if segs_all[g][3] > 0:
            if r not in first_t:
                first_t[r] = g
            last_t[r] = g
        last_g[r] = g

    g_a, g_w, g_row, g_wt, g_wv, g_cv, g_cs, g_tf, g_tl, g_rl = \
        [], [], [], [], [], [], [], [], [], []
    junk_w = 1
    for g in range(ng_tot):
        r, w, a, wt, wv = segs_all[g]
        g_a.append(int(a))
        g_w.append(int(w))
        g_row.append(int(r))
        g_wt.append(int(wt))
        g_wv.append(int(wv))
        g_cv.append(int(r * SMv + cv_slot[g]) if cv_slot[g] >= 0 else -1)
        g_cs.append(int(r * SMs + cs_slot[g]) if cs_slot[g] >= 0 else -1)
        g_tf.append(1 if first_t.get(r, -1) == g else 0)
        g_tl.append(1 if last_t.get(r, -1) == g else 0)
        g_rl.append(1 if last_g[r] == g else 0)
        junk_w = max(junk_w, int(w - wt - wv))

    r_tcol, r_vc0, r_vns, r_sc0, r_sns = [], [], [], [], []
    for r in range(RP):
        r_tcol.append(int(r * SMv + tslot[r]) if tslot[r] >= 0 else -1)
        r_vc0.append(int(r * SMv))
        r_vns.append(0 if v_direct[r] else int(vcnt[r]))
        r_sc0.append(int(r * SMs))
        r_sns.append(0 if s_direct[r] else int(scnt[r]))

    use_t_any = any(has_t)
    junk32_w = max(1, SMs)

    cfg = {
        "lnc": int(lnc),
        "mode": "rows" if rows_mode else "chunks",
        "rows_per_partition": int(RP),
        "n_units": int(nu),
        "unit_elems": [int(v) for v in u_len],
        "prefetch": PREFETCH,
        "main_bytes": MAIN_BYTES,
        "last_main_bytes": LAST_MAIN_BYTES,
        "tail_bytes": list(TAIL_BYTES),
        "use_tensor_engine": bool(use_t_any),
        "mm_width": MM_W,
        "no_t_tail_units": NO_T_TAIL,
        "v_slots": int(SMv),
        "s_slots": int(SMs),
    }
    args = (tuple(u_off), tuple(u_len), tuple(u_g0), tuple(u_ng),
            tuple(g_a), tuple(g_w), tuple(g_row), tuple(g_wt), tuple(g_wv),
            tuple(g_cv), tuple(g_cs), tuple(g_tf), tuple(g_tl), tuple(g_rl),
            tuple(r_tcol), tuple(r_vc0), tuple(r_vns), tuple(r_sc0), tuple(r_sns),
            int(nu), int(RP * SMv), int(RP * SMs), int(junk_w), int(junk32_w),
            bool(use_t_any))
    return cfg, args


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    global _LAST_CONFIG
    if x.ndim != 2:
        raise ValueError("expected a 2-D input")
    d = dim
    if d < 0:
        d = d + 2
    if d != 1:
        raise ValueError("only dim=1 is supported")
    M = int(x.shape[0])
    N = int(x.shape[1])
    cfg, args = _plan(M, N, x.dtype)
    inv_n = 1.0 / float(N)
    kern = _row_mean_kernel[cfg["lnc"]]
    out = kern(x, *args, inv_n)
    _LAST_CONFIG = dict(cfg)
    return out.view(M)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
