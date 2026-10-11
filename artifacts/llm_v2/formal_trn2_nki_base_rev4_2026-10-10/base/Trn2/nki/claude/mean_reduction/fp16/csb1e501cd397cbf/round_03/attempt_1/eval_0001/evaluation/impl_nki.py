import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (deterministic functions of the input shape only)
# ----------------------------------------------------------------------------
P = 128                                  # partitions per tile
PREFETCH = 5                             # DMA units issued ahead of compute
MAIN_BYTES = 16384                       # bytes / partition per main DMA unit (2 MiB)
TAIL_BYTES = (1024, 2048, 4096, 8192)    # final units (from the very end backwards)
ALIGN_BYTES = 1024                       # unit sizes are multiples of this (per partition)
T_MIN_N = 1024                           # TensorE identity-accumulation for N >= this
T_RATE_SMALL = 1.6                       # assumed TensorE elem/ns (256-wide accumulator)
T_RATE_LARGE = 1.9                       # assumed TensorE elem/ns (512-wide accumulator)
VS_RATE = 2.1                            # assumed VectorE+ScalarE combined elem/ns
GRAN = 64                                # V/S split granularity (elements)

_LAST_CONFIG = {}


@nki.jit
def _row_mean_kernel(x, u_off, u_len, u_s0, u_s1,
                     s_loc, s_row, s_wt, s_wv, s_ws, s_cv, s_cs,
                     s_tfirst, s_tlast, s_rlast,
                     r_tcol, r_vc0, r_vns, r_sc0, r_sns,
                     n_units, nv_cols, ns_cols, junk_w, junk32_w,
                     mm_w, use_t, inv_n):
    # Program pid owns rows [pid*P*RP, (pid+1)*P*RP); partition p holds rows
    # pid*P*RP + p*RP + r (r in [0, RP)), contiguous in HBM -> one flat stream
    # of RP*N elements per partition, cut into DMA units.
    M = x.shape[0]
    N = x.shape[1]
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    RP = M // (n_prog * P)

    out = nl.ndarray((M, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    x2 = x.reshape((n_prog, P, RP * N))
    ov = out.reshape((n_prog, P, RP))

    ident = None
    if use_t:
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

    # VectorE-owned partial slots / row totals, ScalarE-owned partial slots / row totals
    vparts = nl.ndarray((P, nv_cols), dtype=nl.float32, buffer=nl.sbuf)
    sparts = nl.ndarray((P, ns_cols), dtype=nl.float32, buffer=nl.sbuf)
    vt = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    st = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    # write-only scratch for ScalarE (only the fused fp32 sums are used)
    junk = nl.ndarray((P, junk_w), dtype=x.dtype, buffer=nl.sbuf)
    junk32 = nl.ndarray((P, junk32_w), dtype=nl.float32, buffer=nl.sbuf)

    b1 = None
    b2 = None
    b3 = None
    b4 = None
    b5 = None
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
        if it >= PREFETCH:
            u = it - PREFETCH
            t = b1
            # deferred VectorE work left over from the previous unit
            if pt_on:
                nisa.tensor_reduce(dst=vparts[:, pt_col:pt_col + 1], op=nl.add,
                                   data=pt_acc, axis=1)
                pt_on = False
            if pv_on:
                vc0 = r_vc0[pv_row]
                vns = r_vns[pv_row]
                nisa.tensor_reduce(dst=vt[:, pv_row:pv_row + 1], op=nl.add,
                                   data=vparts[:, vc0:vc0 + vns], axis=1)
                pv_on = False
            for si in range(u_s0[u], u_s1[u]):
                loc = s_loc[si]
                r = s_row[si]
                wt = s_wt[si]
                wv = s_wv[si]
                ws = s_ws[si]
                # TensorE: identity matmul, fp32 PSUM accumulation per row
                if wt > 0:
                    if s_tfirst[si] > 0:
                        acc = nl.ndarray((P, mm_w), dtype=nl.float32, buffer=nl.psum)
                        n_mm = 0
                    for jj in range(wt // mm_w):
                        c0 = loc + jj * mm_w
                        nisa.nc_matmul(dst=acc, stationary=ident,
                                       moving=t[:, c0:c0 + mm_w],
                                       accumulate=(n_mm > 0))
                        n_mm = n_mm + 1
                # VectorE raw part
                a0 = loc + wt
                if wv > 0:
                    cv = s_cv[si]
                    nisa.tensor_reduce(dst=vparts[:, cv:cv + 1], op=nl.add,
                                       data=t[:, a0:a0 + wv], axis=1)
                # ScalarE raw part
                a1 = a0 + wv
                if ws > 0:
                    cs = s_cs[si]
                    nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                    data=t[:, a1:a1 + ws], reduce_op=nl.add,
                                    reduce_res=sparts[:, cs:cs + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                # ScalarE row total (depends only on ScalarE writes)
                if s_rlast[si] > 0:
                    sc0 = r_sc0[r]
                    sns = r_sns[r]
                    nisa.activation(dst=junk32[:, 0:sns], op=nl.copy,
                                    data=sparts[:, sc0:sc0 + sns], reduce_op=nl.add,
                                    reduce_res=st[:, r:r + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                # deferred VectorE work from the previous segment
                if pt_on:
                    nisa.tensor_reduce(dst=vparts[:, pt_col:pt_col + 1], op=nl.add,
                                       data=pt_acc, axis=1)
                    pt_on = False
                if pv_on:
                    vc0 = r_vc0[pv_row]
                    vns = r_vns[pv_row]
                    nisa.tensor_reduce(dst=vt[:, pv_row:pv_row + 1], op=nl.add,
                                       data=vparts[:, vc0:vc0 + vns], axis=1)
                    pv_on = False
                if s_tlast[si] > 0:
                    pt_on = True
                    pt_acc = acc
                    pt_col = r_tcol[r]
                if s_rlast[si] > 0:
                    if s_tlast[si] > 0:
                        pv_on = True
                        pv_row = r
                    else:
                        vc0 = r_vc0[r]
                        vns = r_vns[r]
                        nisa.tensor_reduce(dst=vt[:, r:r + 1], op=nl.add,
                                           data=vparts[:, vc0:vc0 + vns], axis=1)
        b1 = b2
        b2 = b3
        b3 = b4
        b4 = b5
        b5 = tn

    if pt_on:
        nisa.tensor_reduce(dst=vparts[:, pt_col:pt_col + 1], op=nl.add,
                           data=pt_acc, axis=1)
    if pv_on:
        vc0 = r_vc0[pv_row]
        vns = r_vns[pv_row]
        nisa.tensor_reduce(dst=vt[:, pv_row:pv_row + 1], op=nl.add,
                           data=vparts[:, vc0:vc0 + vns], axis=1)

    tot = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=vt, data2=st, op=nl.add)
    res = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res, data=tot, op0=nl.multiply, operand0=inv_n)
    nisa.dma_copy(dst=ov[pid], src=res)
    return out


def _plan(M, N, dtype):
    two_byte = dtype in (torch.float16, torch.bfloat16)
    itemsize = 2 if two_byte else 4
    if M % (2 * P) == 0:
        lnc = 2
    elif M % P == 0:
        lnc = 1
    else:
        raise ValueError("mean_reduction kernel requires M to be a multiple of 128")
    RP = M // (lnc * P)
    L = RP * N
    align = ALIGN_BYTES // itemsize
    main = MAIN_BYTES // itemsize

    # ---- DMA units over the flat per-partition stream [0, L) ----
    tail = []
    rem = L
    for tb in TAIL_BYTES:
        te = tb // itemsize
        if rem >= te:
            tail.append(te)
            rem -= te
        else:
            break
    front = []
    if rem > 0:
        nb = rem // align
        extra = rem - nb * align
        if nb == 0:
            front.append(rem)
        else:
            nf = (rem + main - 1) // main
            nf = max(1, min(nf, nb))
            q, rr = divmod(nb, nf)
            for i in range(nf):
                front.append((q + (1 if i < rr else 0)) * align)
            front[0] += extra
    sizes = front + tail[::-1]
    nu = len(sizes)
    offs = []
    o = 0
    for s in sizes:
        offs.append(o)
        o += s

    # ---- engine split ----
    use_t = bool(two_byte and N >= T_MIN_N)
    W = 256 if N <= 2048 else 512
    rt = T_RATE_SMALL if W == 256 else T_RATE_LARGE
    ft = (N + W) * rt / (rt + VS_RATE) / float(N)
    ft = min(ft, 0.85)

    segs = []   # [unit, loc, row, w, wt, wv, ws]
    for u in range(nu):
        u0 = offs[u]
        u1 = u0 + sizes[u]
        r = u0 // N
        while r * N < u1:
            a = max(u0, r * N)
            b = min(u1, (r + 1) * N)
            w = b - a
            wt = 0
            if use_t and u < nu - 1 and w > 256:
                wt = int(round(w * ft / W)) * W
                cap = ((w - 256) // W) * W
                wt = max(0, min(wt, cap))
            segs.append([u, a - u0, r, w, wt, 0, 0])
            r += 1

    t_rows = set()
    tot_rem = 0
    for s in segs:
        if s[4] > 0:
            t_rows.add(s[2])
        tot_rem += s[3] - s[4]
    extra_v = W * len(t_rows)
    fv = 0.5
    if tot_rem > 0:
        fv = (tot_rem - extra_v) / (2.0 * tot_rem)
    fv = max(0.2, min(0.5, fv))
    for s in segs:
        rem_ = s[3] - s[4]
        if rem_ >= 2 * GRAN:
            wv = int(round(rem_ * fv / GRAN)) * GRAN
            wv = max(GRAN, min(rem_ - GRAN, wv))
        else:
            wv = max(1, rem_ // 2)
        s[5] = wv
        s[6] = rem_ - wv

    nseg = len(segs)
    first_t = {}
    last_t = {}
    last_seg = {}
    for i in range(nseg):
        r = segs[i][2]
        if segs[i][4] > 0:
            if r not in first_t:
                first_t[r] = i
            last_t[r] = i
        last_seg[r] = i

    vcnt = [0] * RP
    scnt = [0] * RP
    cv_slot = [-1] * nseg
    cs_slot = [-1] * nseg
    for i in range(nseg):
        r = segs[i][2]
        if segs[i][5] > 0:
            cv_slot[i] = vcnt[r]
            vcnt[r] += 1
        if segs[i][6] > 0:
            cs_slot[i] = scnt[r]
            scnt[r] += 1
    tslot = [-1] * RP
    for r in sorted(last_t.keys()):
        tslot[r] = vcnt[r]
        vcnt[r] += 1
    SMv = max(1, max(vcnt))
    SMs = max(1, max(scnt))

    u_s0 = [0] * nu
    u_s1 = [0] * nu
    for i in range(nseg):
        u = segs[i][0]
        if i == 0 or segs[i - 1][0] != u:
            u_s0[u] = i
        u_s1[u] = i + 1

    s_loc, s_row, s_wt, s_wv, s_ws, s_cv, s_cs = [], [], [], [], [], [], []
    s_tf, s_tl, s_rl = [], [], []
    junk_w = 1
    for i in range(nseg):
        u, loc, r, w, wt, wv, ws = segs[i]
        s_loc.append(int(loc))
        s_row.append(int(r))
        s_wt.append(int(wt))
        s_wv.append(int(wv))
        s_ws.append(int(ws))
        s_cv.append(int(r * SMv + cv_slot[i]) if cv_slot[i] >= 0 else -1)
        s_cs.append(int(r * SMs + cs_slot[i]) if cs_slot[i] >= 0 else -1)
        s_tf.append(1 if first_t.get(r, -1) == i else 0)
        s_tl.append(1 if last_t.get(r, -1) == i else 0)
        s_rl.append(1 if last_seg[r] == i else 0)
        junk_w = max(junk_w, int(ws))

    r_tcol = []
    r_vc0 = []
    r_vns = []
    r_sc0 = []
    r_sns = []
    for r in range(RP):
        r_tcol.append(int(r * SMv + tslot[r]) if tslot[r] >= 0 else -1)
        r_vc0.append(int(r * SMv))
        r_vns.append(int(vcnt[r]))
        r_sc0.append(int(r * SMs))
        r_sns.append(int(scnt[r]))

    use_t_any = len(t_rows) > 0
    junk32_w = max(1, SMs)

    cfg = {
        "lnc": int(lnc),
        "rows_per_partition": int(RP),
        "unit_elems": [int(v) for v in sizes],
        "n_units": int(nu),
        "n_segments": int(nseg),
        "prefetch": PREFETCH,
        "main_bytes": MAIN_BYTES,
        "tail_bytes": list(TAIL_BYTES),
        "use_tensor_engine": bool(use_t_any),
        "mm_width": int(W),
        "t_frac": float(round(ft, 4)),
        "v_frac": float(round(fv, 4)),
        "v_slots": int(SMv),
        "s_slots": int(SMs),
    }
    args = (tuple(offs), tuple(int(v) for v in sizes), tuple(u_s0), tuple(u_s1),
            tuple(s_loc), tuple(s_row), tuple(s_wt), tuple(s_wv), tuple(s_ws),
            tuple(s_cv), tuple(s_cs), tuple(s_tf), tuple(s_tl), tuple(s_rl),
            tuple(r_tcol), tuple(r_vc0), tuple(r_vns), tuple(r_sc0), tuple(r_sns),
            int(nu), int(RP * SMv), int(RP * SMs), int(junk_w), int(junk32_w),
            int(W), bool(use_t_any))
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
