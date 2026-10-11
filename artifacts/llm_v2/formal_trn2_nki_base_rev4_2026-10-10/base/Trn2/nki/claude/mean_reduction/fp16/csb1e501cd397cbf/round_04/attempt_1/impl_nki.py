import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (deterministic functions of the input shape only)
# ----------------------------------------------------------------------------
P = 128                               # partitions per tile
PREFETCH = 6                          # DMA units issued ahead of compute
MAIN_BYTES = 8192                     # bytes / partition of a regular DMA unit (1 MiB)
HEAD_BYTES = 4096                     # first unit of the first row is at most this
TAIL_BYTES = (1024, 2048, 4096)       # final units of the last row (from the end)
ALIGN_BYTES = 1024                    # chunk sizes are multiples of this (per partition)
T_MIN_N = 2048                        # TensorE identity-accumulation for N >= this
W_WIDE = 512                          # PSUM accumulator width for N >= 4096
W_NARROW = 256                        # accumulator width for smaller N and the last row
GRAN = 64                             # V/S split granularity (elements)

_LAST_CONFIG = {}


@nki.jit
def _row_mean_kernel(x, u_off, u_len, u_row, u_wt, u_wv, u_cv, u_cs,
                     u_tfirst, u_tlast, u_rlast,
                     r_w, r_tcol, r_vc0, r_vns, r_sc0, r_sns,
                     n_units, nv_cols, ns_cols, junk_w, junk32_w,
                     use_t, inv_n):
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

    # VectorE-owned partial slots / row totals, ScalarE-owned partial slots / row totals
    vparts = nl.ndarray((P, nv_cols), dtype=nl.float32, buffer=nl.sbuf)
    sparts = nl.ndarray((P, ns_cols), dtype=nl.float32, buffer=nl.sbuf)
    vt = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    st = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    # write-only fp32 scratch for ScalarE (only the fused fp32 sums are used)
    junk = nl.ndarray((P, junk_w), dtype=nl.float32, buffer=nl.sbuf)
    junk32 = nl.ndarray((P, junk32_w), dtype=nl.float32, buffer=nl.sbuf)

    ident = None
    b1 = None
    b2 = None
    b3 = None
    b4 = None
    b5 = None
    b6 = None
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
            r = u_row[u]
            w = u_len[u]
            wt = u_wt[u]
            wv = u_wv[u]
            ws = w - wt - wv
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
            # TensorE: identity matmul, fp32 PSUM accumulation per row
            if wt > 0:
                rw = r_w[r]
                if u_tfirst[u] > 0:
                    acc = nl.ndarray((P, rw), dtype=nl.float32, buffer=nl.psum)
                    n_mm = 0
                for jj in range(wt // rw):
                    nisa.nc_matmul(dst=acc, stationary=ident,
                                   moving=t[:, jj * rw:(jj + 1) * rw],
                                   accumulate=(n_mm > 0))
                    n_mm = n_mm + 1
            # VectorE raw part
            cv = u_cv[u]
            if wv > 0:
                if cv >= 0:
                    nisa.tensor_reduce(dst=vparts[:, cv:cv + 1], op=nl.add,
                                       data=t[:, wt:wt + wv], axis=1)
                else:
                    nisa.tensor_reduce(dst=vt[:, r:r + 1], op=nl.add,
                                       data=t[:, wt:wt + wv], axis=1)
            # ScalarE raw part
            cs = u_cs[u]
            a1 = wt + wv
            if ws > 0:
                if cs >= 0:
                    nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                    data=t[:, a1:w], reduce_op=nl.add,
                                    reduce_res=sparts[:, cs:cs + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                else:
                    nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                    data=t[:, a1:w], reduce_op=nl.add,
                                    reduce_res=st[:, r:r + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
            # ScalarE row total (rows other than the last one)
            if u_rlast[u] > 0:
                sns = r_sns[r]
                if sns > 0:
                    if r < RP - 1:
                        sc0 = r_sc0[r]
                        nisa.activation(dst=junk32[:, 0:sns], op=nl.copy,
                                        data=sparts[:, sc0:sc0 + sns],
                                        reduce_op=nl.add,
                                        reduce_res=st[:, r:r + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
            if u_tlast[u] > 0:
                pt_on = True
                pt_acc = acc
                pt_col = r_tcol[r]
            if u_rlast[u] > 0:
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
        b4 = b5
        b5 = b6
        b6 = tn

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


def _head_chunks(n, main, align, head):
    if n <= head:
        return [n]
    rest = n - head
    if rest < align:
        return _even_chunks(n, main, align)
    return [head] + _even_chunks(rest, main, align)


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
    itemsize = 2 if two_byte else 4
    if M % (2 * P) == 0:
        lnc = 2
    elif M % P == 0:
        lnc = 1
    else:
        raise ValueError("mean_reduction kernel requires M to be a multiple of 128")
    RP = M // (lnc * P)
    main = MAIN_BYTES // itemsize
    head = HEAD_BYTES // itemsize
    align = ALIGN_BYTES // itemsize
    tails = [b // itemsize for b in TAIL_BYTES]
    use_t = bool(two_byte and N >= T_MIN_N)
    w_main = W_WIDE if N >= 4096 else W_NARROW

    # ---- row-aligned DMA units (each unit = one chunk of one row) ----
    units = []
    for r in range(RP):
        if r == RP - 1:
            ch = _tail_chunks(N, main, align, tails)
        elif r == 0:
            ch = _head_chunks(N, main, align, head)
        else:
            ch = _even_chunks(N, main, align)
        c0 = 0
        for w in ch:
            units.append((r, c0, w))
            c0 += w
    nu = len(units)
    r_w = []
    for r in range(RP):
        r_w.append(W_NARROW if r == RP - 1 else w_main)

    # ---- TensorE share ----
    wts = []
    for u in range(nu):
        r, c0, w = units[u]
        wt = 0
        if use_t and u < nu - 1:
            W = r_w[r]
            if w >= 2 * W:
                wt = ((w // 2) // W) * W
                cap = ((w - 256) // W) * W
                wt = max(0, min(wt, cap))
        wts.append(wt)
    row_wt = [0] * RP
    for u in range(nu):
        row_wt[units[u][0]] += wts[u]

    # ---- VectorE / ScalarE split of the remainder ----
    wvs = []
    for u in range(nu):
        r, c0, w = units[u]
        wt = wts[u]
        rem = w - wt
        extra = 0
        if wt > 0 and row_wt[r] > 0:
            extra = (r_w[r] * wt) // row_wt[r]
        if rem >= 2 * GRAN:
            wv = (((rem - extra) // 2) // GRAN) * GRAN
            wv = max(GRAN, min(rem - GRAN, wv))
        else:
            wv = max(1, rem // 2)
        wvs.append(wv)

    has_t = [False] * RP
    vseg = [0] * RP
    sseg = [0] * RP
    for u in range(nu):
        r = units[u][0]
        w = units[u][2]
        if wts[u] > 0:
            has_t[r] = True
        if wvs[u] > 0:
            vseg[r] += 1
        if w - wts[u] - wvs[u] > 0:
            sseg[r] += 1
    for r in range(RP):
        if sseg[r] == 0 or (vseg[r] == 0 and not has_t[r]):
            raise ValueError("unsupported split")
    v_direct = []
    s_direct = []
    for r in range(RP):
        v_direct.append((not has_t[r]) and vseg[r] == 1)
        s_direct.append(sseg[r] == 1)

    vcnt = [0] * RP
    scnt = [0] * RP
    cv_slot = [-1] * nu
    cs_slot = [-1] * nu
    for u in range(nu):
        r = units[u][0]
        w = units[u][2]
        if wvs[u] > 0 and not v_direct[r]:
            cv_slot[u] = vcnt[r]
            vcnt[r] += 1
        if (w - wts[u] - wvs[u]) > 0 and not s_direct[r]:
            cs_slot[u] = scnt[r]
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
    last_u = {}
    for u in range(nu):
        r = units[u][0]
        if wts[u] > 0:
            if r not in first_t:
                first_t[r] = u
            last_t[r] = u
        last_u[r] = u

    u_off, u_len, u_row, u_wt, u_wv, u_cv, u_cs = [], [], [], [], [], [], []
    u_tf, u_tl, u_rl = [], [], []
    junk_w = 1
    for u in range(nu):
        r, c0, w = units[u]
        u_off.append(int(r * N + c0))
        u_len.append(int(w))
        u_row.append(int(r))
        u_wt.append(int(wts[u]))
        u_wv.append(int(wvs[u]))
        u_cv.append(int(r * SMv + cv_slot[u]) if cv_slot[u] >= 0 else -1)
        u_cs.append(int(r * SMs + cs_slot[u]) if cs_slot[u] >= 0 else -1)
        u_tf.append(1 if first_t.get(r, -1) == u else 0)
        u_tl.append(1 if last_t.get(r, -1) == u else 0)
        u_rl.append(1 if last_u[r] == u else 0)
        junk_w = max(junk_w, int(w - wts[u] - wvs[u]))

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
        "rows_per_partition": int(RP),
        "n_units": int(nu),
        "unit_elems": [int(v) for v in u_len],
        "prefetch": PREFETCH,
        "main_bytes": MAIN_BYTES,
        "head_bytes": HEAD_BYTES,
        "tail_bytes": list(TAIL_BYTES),
        "use_tensor_engine": bool(use_t_any),
        "mm_width_main": int(w_main),
        "mm_width_last": int(W_NARROW),
        "v_slots": int(SMv),
        "s_slots": int(SMs),
    }
    args = (tuple(u_off), tuple(u_len), tuple(u_row), tuple(u_wt), tuple(u_wv),
            tuple(u_cv), tuple(u_cs), tuple(u_tf), tuple(u_tl), tuple(u_rl),
            tuple(int(v) for v in r_w), tuple(r_tcol), tuple(r_vc0), tuple(r_vns),
            tuple(r_sc0), tuple(r_sns),
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
