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
    # G: number of 128-row tiles packed into one transpose / matmul (small D only)
    if D <= 64:
        G = P // D
    else:
        G = 1
    if G > 1:
        DC = 1
    else:
        DC = (D + P - 1) // P
    DA = D + 1
    UPO = BANK // (G * DA)           # transpose-units per output PSUM bank
    if UPO < 1:
        UPO = 1
    if DC == 1:
        NBU = 4                      # transpose-units per PSUM batch
    else:
        NBU = 2
    unit_rows = NBU * G
    T_full = (M // 2) // P
    TB = ((1024 // D) // unit_rows) * unit_rows
    if TB < unit_rows:
        TB = unit_rows
    tb_par = (((T_full + 3) // 4) // unit_rows) * unit_rows
    if tb_par < unit_rows:
        tb_par = unit_rows
    if tb_par < TB:
        TB = tb_par
    return {"G": G, "DC": DC, "NBU": NBU, "UPO": UPO, "TB": TB,
            "lnc_split": 2, "matmul_dtype": "float32"}


def _phi(dst, src, e_t, u_t):
    # phi(x) = min(exp(x), max(x, 0) + 1)  (== x+1 for x>0, exp(x) otherwise)
    nisa.activation(dst=e_t, op=nl.exp, data=src)
    nisa.tensor_scalar(dst=u_t, data=src, op0=nl.maximum, operand0=0.0,
                       op1=nl.add, operand1=1.0, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=dst, data1=e_t, data2=u_t, op=nl.minimum,
                       engine=nisa.engine.vector)


def _state_rows(s_ps, k_t, v_t, ones, nrows, nt, D, DC, idx0):
    # k_t: [nrows, nt*D] raw K rows (overwritten by phi(K)), v_t: [nrows, nt*D] V rows
    n = nt * D
    e_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    u_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
    _phi(k_t[0:nrows, 0:n], k_t[0:nrows, 0:n],
         e_t[0:nrows, 0:n], u_t[0:nrows, 0:n])
    for t in range(nt):
        for c in range(DC):
            c0 = c * P
            dc = min(P, D - c0)
            nisa.nc_matmul(dst=s_ps[0:dc, c * BANK:c * BANK + D],
                           stationary=k_t[0:nrows, t * D + c0:t * D + c0 + dc],
                           moving=v_t[0:nrows, t * D:t * D + D],
                           accumulate=(idx0 + t > 0))
            nisa.nc_matmul(dst=s_ps[0:dc, c * BANK + D:c * BANK + D + 1],
                           stationary=k_t[0:nrows, t * D + c0:t * D + c0 + dc],
                           moving=ones[0:nrows, 0:1],
                           accumulate=(idx0 + t > 0))


def _out_batch(q_t, o_t, nrows, s, ntb, s_mov, D, G, DC, NBU, UPO, eps):
    DA = D + 1
    RW = NBU * P
    if G > 1:
        nu = (ntb + G - 1) // G
    else:
        nu = ntb
    full = 1
    if nrows != P:
        full = 0
    if G > 1:
        if ntb != nu * G:
            full = 0

    # ---- transposes of raw Q tiles into PSUM ----
    pt = nl.ndarray((P, DC * RW), dtype=nl.float32, buffer=nl.psum)
    for u in range(nu):
        if G > 1:
            g = min(G, ntb - u * G)
            t0 = s + u * G
            nisa.nc_transpose(dst=pt[0:g * D, u * P:u * P + nrows],
                              data=q_t[0:nrows, t0 * D:t0 * D + g * D],
                              engine=nisa.engine.tensor)
        else:
            t = s + u
            for c in range(DC):
                c0 = c * P
                dc = min(P, D - c0)
                a0 = c * RW + u * P
                nisa.nc_transpose(dst=pt[0:dc, a0:a0 + nrows],
                                  data=q_t[0:nrows, t * D + c0:t * D + c0 + dc],
                                  engine=nisa.engine.tensor)

    # ---- feature map fused with PSUM evacuation ----
    ph = nl.ndarray((P, DC * RW), dtype=nl.float32, buffer=nl.sbuf)
    e_t = nl.ndarray((P, DC * RW), dtype=nl.float32, buffer=nl.sbuf)
    u_t = nl.ndarray((P, DC * RW), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(DC):
        cb = c * RW
        if G > 1:
            wc = G * D
        else:
            wc = min(P, D - c * P)
        if full == 1:
            a1 = cb + nu * P
            _phi(ph[0:wc, cb:a1], pt[0:wc, cb:a1],
                 e_t[0:wc, cb:a1], u_t[0:wc, cb:a1])
        else:
            for u in range(nu):
                if G > 1:
                    w = min(G, ntb - u * G) * D
                else:
                    w = wc
                a0 = cb + u * P
                a1 = a0 + nrows
                _phi(ph[0:w, a0:a1], pt[0:w, a0:a1],
                     e_t[0:w, a0:a1], u_t[0:w, a0:a1])

    # ---- numerator + denominator matmul, then normalisation ----
    n_og = (nu + UPO - 1) // UPO
    for og in range(n_og):
        u0 = og * UPO
        u1 = min(nu, u0 + UPO)
        op = nl.ndarray((P, BANK), dtype=nl.float32, buffer=nl.psum)
        nrt = 0
        for u in range(u0, u1):
            if G > 1:
                g = min(G, ntb - u * G)
            else:
                g = 1
            col = (u - u0) * G * DA
            for c in range(DC):
                a0 = c * RW + u * P
                if G > 1:
                    w = g * D
                    mov = s_mov[0:w, 0:g * DA]
                else:
                    w = min(P, D - c * P)
                    mov = s_mov[0:w, c * DA:c * DA + DA]
                nisa.nc_matmul(dst=op[0:nrows, col:col + g * DA],
                               stationary=ph[0:w, a0:a0 + nrows],
                               moving=mov,
                               accumulate=(c > 0))
            nrt = nrt + g
        den = nl.ndarray((P, UPO * G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=den[0:nrows, 0:nrt],
                           data=op[0:nrows, D:D + (nrt - 1) * DA + 1:DA],
                           op0=nl.add, operand0=eps, engine=nisa.engine.vector)
        rden = nl.ndarray((P, UPO * G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rden[0:nrows, 0:nrt], data=den[0:nrows, 0:nrt])
        for r in range(nrt):
            trow = s + u0 * G + r
            nisa.tensor_scalar(dst=o_t[0:nrows, trow * D:trow * D + D],
                               data=op[0:nrows, r * DA:r * DA + D],
                               op0=nl.multiply, operand0=rden[0:nrows, r:r + 1],
                               engine=nisa.engine.vector)


@nki.jit
def _lsa_kernel(Q, K, V, eps, G, DC, NBU, UPO, TB):
    M = Q.shape[0]
    D = Q.shape[1]
    DA = D + 1
    out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R = M // n_prog
    row0 = pid * R
    T_full = R // P
    rem = R - T_full * P
    nblk = (T_full + TB - 1) // TB

    ones = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    # ================= phase 1: [S | Z] = phi(K)^T [V | 1] (partial per core) =================
    s_ps = nl.ndarray((P, DC * BANK), dtype=nl.float32, buffer=nl.psum)
    for b in range(nblk):
        tb = min(TB, T_full - b * TB)
        base = row0 + b * TB * P
        n = tb * D
        k_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=K.ap(pattern=[[n, P], [1, n]], offset=base * D))
        v_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=V.ap(pattern=[[n, P], [1, n]], offset=base * D))
        _state_rows(s_ps, k_t, v_t, ones, P, tb, D, DC, b * TB)
    if rem > 0:
        base = row0 + T_full * P
        k_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_r[0:rem, :], src=K[base:base + rem, :])
        v_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_r[0:rem, :], src=V[base:base + rem, :])
        _state_rows(s_ps, k_r, v_r, ones, rem, 1, D, DC, T_full)

    # ---- complete reduction over all M rows (combine the two cores) ----
    s_own = nl.ndarray((P, DC * DA), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=s_own, value=0.0)
    for c in range(DC):
        dc = min(P, D - c * P)
        nisa.tensor_copy(dst=s_own[0:dc, c * DA:c * DA + DA],
                         src=s_ps[0:dc, c * BANK:c * BANK + DA],
                         engine=nisa.engine.vector)
    s_tot = nl.ndarray((P, DC * DA), dtype=nl.float32, buffer=nl.sbuf)
    if n_prog > 1:
        other = 1 - pid
        nisa.sendrecv_compute(src=s_own, dst=s_tot, send_to_rank=other,
                              recv_from_rank=other, reduce_op=nl.add)
    else:
        nisa.tensor_copy(dst=s_tot, src=s_own, engine=nisa.engine.vector)

    if G > 1:
        # block-diagonal copy of [S | Z] so G row tiles share one matmul
        s_mov = nl.ndarray((P, G * DA), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=s_mov, value=0.0)
        for j in range(G):
            nisa.dma_copy(dst=s_mov[j * D:j * D + D, j * DA:j * DA + DA],
                          src=s_tot[0:D, 0:DA])
    else:
        s_mov = s_tot

    # ================= phase 2: O = (phi(Q) S) / (phi(Q).Z + eps) =================
    BR = NBU * G
    for b in range(nblk):
        tb = min(TB, T_full - b * TB)
        base = row0 + b * TB * P
        n = tb * D
        q_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=Q.ap(pattern=[[n, P], [1, n]], offset=base * D))
        o_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        for s in range(0, tb, BR):
            ntb = min(BR, tb - s)
            _out_batch(q_t, o_t, P, s, ntb, s_mov, D, G, DC, NBU, UPO, eps)
        nisa.dma_copy(dst=out.ap(pattern=[[n, P], [1, n]], offset=base * D), src=o_t)
    if rem > 0:
        base = row0 + T_full * P
        q_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_r[0:rem, :], src=Q[base:base + rem, :])
        o_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        _out_batch(q_r, o_r, rem, 0, 1, s_mov, D, G, DC, NBU, UPO, eps)
        nisa.dma_copy(dst=out[base:base + rem, :], src=o_r[0:rem, :])

    return out


def run(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, eps: float = 1e-6, **kwargs):
    global _LAST_CONFIG
    M = int(Q.shape[0])
    D = int(Q.shape[1])
    cfg = _make_config(M, D)
    out = _lsa_kernel(Q, K, V, float(eps), cfg["G"], cfg["DC"], cfg["NBU"],
                      cfg["UPO"], cfg["TB"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
