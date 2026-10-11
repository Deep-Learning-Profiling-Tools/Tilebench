import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (deterministic functions of the input shape only)
# ----------------------------------------------------------------------------
P = 128                    # partitions per tile
MM_W = 512                 # matmul moving free size (one fp32 PSUM bank)
PREFETCH = 4               # DMA units issued ahead of compute (program order)
ROW_TARGET_BYTES = 8192    # bytes / partition per DMA unit (1 MiB per DMA)
LAST_TARGET_BYTES = 4096   # smaller units on the final row (short tail)
T_MIN_N = 4096             # use TensorE identity-accumulation when N >= this

_LAST_CONFIG = {}


@nki.jit
def _row_mean_kernel(x, u_r, u_k, u_w, u_c, u_wt, u_wv, u_ci, u_nc, u_rt,
                     n_units, junk_w, nss_max, use_t, inv_n):
    # Program pid owns rows [pid*P*RP, (pid+1)*P*RP); partition p of it holds
    # rows pid*P*RP + p*RP + r (r in [0, RP)), contiguous in HBM.
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

    # VectorE-owned per-row totals and ScalarE-owned per-row totals (fp32)
    vt = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    st = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    # write-only scratch for ScalarE (only the fused fp32 sums are used)
    junk = nl.ndarray((P, junk_w), dtype=x.dtype, buffer=nl.sbuf)
    junk32 = nl.ndarray((P, nss_max), dtype=nl.float32, buffer=nl.sbuf)

    b1 = None
    b2 = None
    b3 = None
    b4 = None
    pv_cur = None
    ps_cur = None
    acc = None
    n_mm = 0
    p_on = False
    p_pv = None
    p_ps = None
    p_acc = None
    p_r = 0
    p_nc = 1
    p_rt = 0

    for it in range(n_units + PREFETCH):
        tn = None
        if it < n_units:
            ln = u_k[it] * u_w[it]
            off = u_r[it] * N + u_c[it]
            tn = nl.ndarray((P, ln), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tn, src=x2[pid, :, off:off + ln])
        if it >= PREFETCH:
            u = it - PREFETCH
            t = b1
            r0 = u_r[u]
            k = u_k[u]
            w = u_w[u]
            wt = u_wt[u]
            wv = u_wv[u]
            ci = u_ci[u]
            nc = u_nc[u]
            rt = u_rt[u]
            ws = w - wt - wv
            if k > 1:
                # several full rows in one unit: V / S split each row
                for j in range(k):
                    base = j * w
                    nisa.tensor_reduce(dst=vt[:, r0 + j:r0 + j + 1], op=nl.add,
                                       data=t[:, base:base + wv], axis=1)
                    nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                    data=t[:, base + wv:base + w],
                                    reduce_op=nl.add,
                                    reduce_res=st[:, r0 + j:r0 + j + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
            else:
                multi_v = False
                if nc > 1:
                    multi_v = True
                if rt > 0:
                    multi_v = True
                if ci == 0:
                    if multi_v:
                        pv_cur = nl.ndarray((P, nc + rt), dtype=nl.float32,
                                            buffer=nl.sbuf)
                    if nc > 1:
                        ps_cur = nl.ndarray((P, nc), dtype=nl.float32,
                                            buffer=nl.sbuf)
                    if rt > 0:
                        acc = nl.ndarray((P, MM_W), dtype=nl.float32,
                                         buffer=nl.psum)
                        n_mm = 0
                # TensorE: identity matmul, fp32 PSUM accumulation per row
                if wt > 0:
                    for jj in range(wt // MM_W):
                        nisa.nc_matmul(dst=acc, stationary=ident,
                                       moving=t[:, jj * MM_W:(jj + 1) * MM_W],
                                       accumulate=(n_mm > 0))
                        n_mm = n_mm + 1
                # VectorE part
                if multi_v:
                    nisa.tensor_reduce(dst=pv_cur[:, ci:ci + 1], op=nl.add,
                                       data=t[:, wt:wt + wv], axis=1)
                else:
                    nisa.tensor_reduce(dst=vt[:, r0:r0 + 1], op=nl.add,
                                       data=t[:, wt:wt + wv], axis=1)
                # deferred VectorE combine of the previous row
                if p_on:
                    if p_rt > 0:
                        nisa.tensor_reduce(dst=p_pv[:, p_nc:p_nc + 1], op=nl.add,
                                           data=p_acc, axis=1)
                    nisa.tensor_reduce(dst=vt[:, p_r:p_r + 1], op=nl.add,
                                       data=p_pv[:, 0:p_nc + p_rt], axis=1)
                # ScalarE part
                if nc > 1:
                    nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                    data=t[:, wt + wv:w], reduce_op=nl.add,
                                    reduce_res=ps_cur[:, ci:ci + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                else:
                    nisa.activation(dst=junk[:, 0:ws], op=nl.copy,
                                    data=t[:, wt + wv:w], reduce_op=nl.add,
                                    reduce_res=st[:, r0:r0 + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                # deferred ScalarE combine of the previous row
                if p_on:
                    if p_nc > 1:
                        nisa.activation(dst=junk32[:, 0:p_nc], op=nl.copy,
                                        data=p_ps[:, 0:p_nc], reduce_op=nl.add,
                                        reduce_res=st[:, p_r:p_r + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                    p_on = False
                if multi_v:
                    if ci == nc - 1:
                        p_on = True
                        p_pv = pv_cur
                        p_ps = ps_cur
                        p_acc = acc
                        p_r = r0
                        p_nc = nc
                        p_rt = rt
        b1 = b2
        b2 = b3
        b3 = b4
        b4 = tn

    if p_on:
        if p_rt > 0:
            nisa.tensor_reduce(dst=p_pv[:, p_nc:p_nc + 1], op=nl.add,
                               data=p_acc, axis=1)
        nisa.tensor_reduce(dst=vt[:, p_r:p_r + 1], op=nl.add,
                           data=p_pv[:, 0:p_nc + p_rt], axis=1)
        if p_nc > 1:
            nisa.activation(dst=junk32[:, 0:p_nc], op=nl.copy,
                            data=p_ps[:, 0:p_nc], reduce_op=nl.add,
                            reduce_res=st[:, p_r:p_r + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)

    tot = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tot, data1=vt, data2=st, op=nl.add)
    res = nl.ndarray((P, RP), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=res, data=tot, op0=nl.multiply, operand0=inv_n)
    nisa.dma_copy(dst=ov[pid], src=res)
    return out


def _chunks(N, target, align):
    n = (N + target - 1) // target
    c = (N + n - 1) // n
    c = ((c + align - 1) // align) * align
    n = (N + c - 1) // c
    return n, c


def _split(w, use_t):
    wt = 0
    if use_t:
        wt = ((w // 2) // MM_W) * MM_W
    rem = w - wt
    wv = ((rem // 2) // 2) * 2
    return wt, wv


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
    tgt = ROW_TARGET_BYTES // itemsize
    tgt_last = LAST_TARGET_BYTES // itemsize
    use_t = bool(two_byte and N >= T_MIN_N)

    k = 1
    while (2 * k) * N <= tgt and (2 * k) <= RP // 2 and RP % (2 * k) == 0:
        k = 2 * k

    ur, uk, uw, uc, uwt, uwv, uci, unc, urt = [], [], [], [], [], [], [], [], []

    def add(r, kk, w, c0, wt, wv, ci, nc, rt):
        ur.append(int(r)); uk.append(int(kk)); uw.append(int(w)); uc.append(int(c0))
        uwt.append(int(wt)); uwv.append(int(wv)); uci.append(int(ci))
        unc.append(int(nc)); urt.append(int(rt))

    n = 1
    c = N
    n2 = 1
    c2 = N
    if k > 1:
        mode = "rows"
        wv = ((N // 2) // 2) * 2
        r = 0
        while r < RP - k:
            add(r, k, N, 0, 0, wv, 0, 1, 0)
            r += k
        for r in range(RP - k, RP):
            add(r, 1, N, 0, 0, wv, 0, 1, 0)
    else:
        mode = "chunks"
        n, c = _chunks(N, tgt, MM_W)
        n2, c2 = _chunks(N, tgt_last, MM_W)
        for r in range(RP):
            if r == RP - 1:
                nn, cc = n2, c2
            else:
                nn, cc = n, c
            segs = []
            rt = 0
            for ci in range(nn):
                c0 = ci * cc
                w = min(cc, N - c0)
                wt, wv = _split(w, use_t)
                if wt > 0:
                    rt = 1
                segs.append((c0, w, wt, wv))
            for ci in range(nn):
                c0, w, wt, wv = segs[ci]
                add(r, 1, w, c0, wt, wv, ci, nn, rt)

    junk_w = 1
    for i in range(len(uw)):
        junk_w = max(junk_w, uw[i] - uwt[i] - uwv[i])
    nss_max = max(unc) if unc else 1
    use_t_any = any(v > 0 for v in urt)

    cfg = {
        "lnc": lnc,
        "mode": mode,
        "rows_per_partition": int(RP),
        "rows_per_unit": int(k),
        "chunk": int(c),
        "n_chunks": int(n),
        "last_row_chunk": int(c2),
        "last_row_n_chunks": int(n2),
        "use_tensor_engine": bool(use_t_any),
        "prefetch": PREFETCH,
        "n_units": len(ur),
        "row_target_bytes": ROW_TARGET_BYTES,
        "last_target_bytes": LAST_TARGET_BYTES,
        "split": "T:1/2 (N>=4096, 512-multiples) V:1/2 of rest S:rest",
    }
    args = (tuple(ur), tuple(uk), tuple(uw), tuple(uc), tuple(uwt), tuple(uwv),
            tuple(uci), tuple(unc), tuple(urt), len(ur), int(junk_w),
            int(nss_max), bool(use_t_any))
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
    out = kern(x, args[0], args[1], args[2], args[3], args[4], args[5],
               args[6], args[7], args[8], args[9], args[10], args[11],
               args[12], inv_n)
    _LAST_CONFIG = dict(cfg)
    return out.view(M)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
