import torch
import nki
import nki.language as nl
import nki.isa as nisa

try:  # registers the "neuron" device if not yet loaded (no device work)
    import torch_neuronx  # noqa: F401
except Exception:  # pragma: no cover
    pass

P = 128      # partitions
BANK = 512   # fp32 elements per PSUM bank (per partition)

_LAST_CONFIG = {}


def _make_config(M, D):
    DC = (D + P - 1) // P
    T_A = M // P
    T_C = (M // 2) // P
    cap = 1024 // D
    if cap < 1:
        cap = 1
    TBA = min(cap, (T_A + 3) // 4)
    if TBA < 1:
        TBA = 1
    TBC = min(cap, (T_C + 3) // 4)
    if TBC < 1:
        TBC = 1
    NB = 4 // DC
    if NB < 1:
        NB = 1
    NG = min(8, BANK // (D + 1))
    if NG < 1:
        NG = 1
    return {"DC": DC, "TBA": TBA, "TBC": TBC, "NB": NB, "NG": NG,
            "lnc": 2, "state": "redundant_per_core",
            "matmul_dtype": "float32", "z_path": "fp32_elementwise_sum"}


def _phi(dst, src, e_t, u_t):
    # phi(x) = min(exp(x), max(x, 0) + 1)  (== x+1 for x>0, exp(x) otherwise)
    nisa.activation(dst=e_t, op=nl.exp, data=src)
    nisa.tensor_scalar(dst=u_t, data=src, op0=nl.maximum, operand0=0.0,
                       op1=nl.add, operand1=1.0, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=dst, data1=e_t, data2=u_t, op=nl.minimum,
                       engine=nisa.engine.vector)


def _q_batch(q_t, ph_all, rows, t0, nb, j0, D, DC, NJ):
    # transpose nb Q sub-tiles into PSUM (chunk-major slots), apply phi on evacuation
    pt = nl.ndarray((P, BANK), dtype=nl.float32, buffer=nl.psum)
    for i in range(nb):
        t = t0 + i
        for c in range(DC):
            c0 = c * P
            dc = min(P, D - c0)
            col = (c * nb + i) * P
            nisa.nc_transpose(dst=pt[0:dc, col:col + rows],
                              data=q_t[0:rows, t * D + c0:t * D + c0 + dc],
                              engine=nisa.engine.tensor)
    e_t = nl.ndarray((P, BANK), dtype=nl.float32, buffer=nl.sbuf)
    u_t = nl.ndarray((P, BANK), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(DC):
        dc = min(P, D - c * P)
        cb = c * nb * P
        pb = (c * NJ + j0) * P
        if rows == P:
            w = nb * P
            _phi(ph_all[0:dc, pb:pb + w], pt[0:dc, cb:cb + w],
                 e_t[0:dc, cb:cb + w], u_t[0:dc, cb:cb + w])
        else:
            for i in range(nb):
                a = cb + i * P
                q0 = pb + i * P
                _phi(ph_all[0:dc, q0:q0 + rows], pt[0:dc, a:a + rows],
                     e_t[0:dc, a:a + rows], u_t[0:dc, a:a + rows])


def _out_group(ph_all, o_t, s_sb, rows, t0, ng, j0, D, DC, NJ, eps):
    SW = D + 1
    op = nl.ndarray((P, BANK), dtype=nl.float32, buffer=nl.psum)
    for i in range(ng):
        j = j0 + i
        for c in range(DC):
            dc = min(P, D - c * P)
            a0 = (c * NJ + j) * P
            nisa.nc_matmul(dst=op[0:rows, i * SW:i * SW + SW],
                           stationary=ph_all[0:dc, a0:a0 + rows],
                           moving=s_sb[0:dc, c * SW:c * SW + SW],
                           accumulate=(c > 0))
    den = nl.ndarray((P, ng), dtype=nl.float32, buffer=nl.sbuf)
    if ng == 1:
        dsrc = op[0:rows, D:D + 1]
    else:
        dsrc = op[0:rows, D:D + (ng - 1) * SW + 1:SW]
    nisa.tensor_scalar(dst=den[0:rows, 0:ng], data=dsrc,
                       op0=nl.add, operand0=eps, engine=nisa.engine.vector)
    rden = nl.ndarray((P, ng), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden[0:rows, 0:ng], data=den[0:rows, 0:ng])
    for i in range(ng):
        t = t0 + i
        nisa.tensor_scalar(dst=o_t[0:rows, t * D:t * D + D],
                           data=op[0:rows, i * SW:i * SW + D],
                           op0=nl.multiply, operand0=rden[0:rows, i:i + 1],
                           engine=nisa.engine.vector)


@nki.jit
def _lsa_kernel(Q, K, V, eps, TBA, TBC, NB, NG):
    M = Q.shape[0]
    D = Q.shape[1]
    DC = (D + P - 1) // P
    SW = D + 1
    out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    # ---- state geometry: all M rows (computed redundantly on each core) ----
    T_A = M // P
    remA = M - T_A * P
    nA = (T_A + TBA - 1) // TBA
    tbA0 = min(TBA, T_A)

    # ---- output geometry: this core's half of the rows ----
    R = M // n_prog
    row0 = pid * R
    T_C = R // P
    remC = R - T_C * P
    nC = (T_C + TBC - 1) // TBC
    NJ = T_C
    if remC > 0:
        NJ = T_C + 1

    ph_all = nl.ndarray((P, DC * NJ * P), dtype=nl.float32, buffer=nl.sbuf)
    s_ps = nl.ndarray((P, DC * BANK), dtype=nl.float32, buffer=nl.psum)
    zpart = nl.ndarray((P, tbA0 * D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zpart, value=0.0)

    nloop = max(nA, nC)
    for b in range(nloop):
        if b < nA:
            tb = min(TBA, T_A - b * TBA)
            base = b * TBA * P
            n = tb * D
            k_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=k_t, src=K.ap(pattern=[[n, P], [1, n]], offset=base * D))
            v_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=v_t, src=V.ap(pattern=[[n, P], [1, n]], offset=base * D))
            pk = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
            uk = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
            _phi(pk, k_t, pk, uk)
            # Z partial sums (exact fp32 elementwise accumulation)
            nisa.tensor_tensor(dst=zpart[0:P, 0:n], data1=zpart[0:P, 0:n],
                               data2=pk, op=nl.add, engine=nisa.engine.gpsimd)
            for t in range(tb):
                g = b * TBA + t
                for c in range(DC):
                    c0 = c * P
                    dc = min(P, D - c0)
                    nisa.nc_matmul(dst=s_ps[0:dc, c * BANK:c * BANK + D],
                                   stationary=pk[0:P, t * D + c0:t * D + c0 + dc],
                                   moving=v_t[0:P, t * D:t * D + D],
                                   accumulate=(g > 0))
        if b < nC:
            tbq = min(TBC, T_C - b * TBC)
            baseq = row0 + b * TBC * P
            nq = tbq * D
            q_t = nl.ndarray((P, nq), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=q_t, src=Q.ap(pattern=[[nq, P], [1, nq]], offset=baseq * D))
            for s in range(0, tbq, NB):
                nb = min(NB, tbq - s)
                _q_batch(q_t, ph_all, P, s, nb, b * TBC + s, D, DC, NJ)

    # ---- remainder rows of the state pass ----
    if remA > 0:
        base_r = T_A * P
        k_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        v_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        pk_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        uk_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_r[0:remA, :], src=K[base_r:base_r + remA, :])
        nisa.dma_copy(dst=v_r[0:remA, :], src=V[base_r:base_r + remA, :])
        _phi(pk_r[0:remA, :], k_r[0:remA, :], pk_r[0:remA, :], uk_r[0:remA, :])
        nisa.tensor_tensor(dst=zpart[0:remA, 0:D], data1=zpart[0:remA, 0:D],
                           data2=pk_r[0:remA, 0:D], op=nl.add,
                           engine=nisa.engine.gpsimd)
        for c in range(DC):
            c0 = c * P
            dc = min(P, D - c0)
            nisa.nc_matmul(dst=s_ps[0:dc, c * BANK:c * BANK + D],
                           stationary=pk_r[0:remA, c0:c0 + dc],
                           moving=v_r[0:remA, 0:D],
                           accumulate=True)

    # ---- remainder rows of this core's Q ----
    if remC > 0:
        base_rc = row0 + T_C * P
        q_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_r[0:remC, :], src=Q[base_rc:base_rc + remC, :])
        _q_batch(q_r, ph_all, remC, 0, 1, T_C, D, DC, NJ)

    # ---- fold Z partial-sum segments (fixed tree order) ----
    cnt = tbA0
    while cnt > 1:
        h = cnt // 2
        nisa.tensor_tensor(dst=zpart[0:P, 0:h * D], data1=zpart[0:P, 0:h * D],
                           data2=zpart[0:P, (cnt - h) * D:cnt * D], op=nl.add,
                           engine=nisa.engine.gpsimd)
        cnt = cnt - h

    # ---- moving operand [S | Z] per D-chunk ----
    s_sb = nl.ndarray((P, DC * SW), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(DC):
        c0 = c * P
        dc = min(P, D - c0)
        nisa.tensor_copy(dst=s_sb[0:dc, c * SW:c * SW + D],
                         src=s_ps[0:dc, c * BANK:c * BANK + D],
                         engine=nisa.engine.vector)
        zt = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=zt[0:dc, 0:P], data=zpart[0:P, c0:c0 + dc],
                          engine=nisa.engine.tensor)
        nisa.tensor_reduce(dst=s_sb[0:dc, c * SW + D:c * SW + D + 1], op=nl.add,
                           data=zt[0:dc, 0:P], axis=1)

    # ---- output: O = (phi(Q) S) / (phi(Q).Z + eps) for this core's rows ----
    for b in range(nC):
        tbq = min(TBC, T_C - b * TBC)
        baseq = row0 + b * TBC * P
        nq = tbq * D
        o_t = nl.ndarray((P, nq), dtype=nl.float32, buffer=nl.sbuf)
        for s in range(0, tbq, NG):
            ng = min(NG, tbq - s)
            _out_group(ph_all, o_t, s_sb, P, s, ng, b * TBC + s, D, DC, NJ, eps)
        nisa.dma_copy(dst=out.ap(pattern=[[nq, P], [1, nq]], offset=baseq * D), src=o_t)
    if remC > 0:
        base_rc2 = row0 + T_C * P
        o_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        _out_group(ph_all, o_r, s_sb, remC, 0, 1, T_C, D, DC, NJ, eps)
        nisa.dma_copy(dst=out[base_rc2:base_rc2 + remC, :], src=o_r[0:remC, :])

    return out


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6, **kwargs):
    global _LAST_CONFIG
    M = int(Q.shape[0])
    D = int(Q.shape[1])
    cfg = _make_config(M, D)
    out = _lsa_kernel(Q, K, V, float(eps), cfg["TBA"], cfg["TBC"], cfg["NB"], cfg["NG"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
