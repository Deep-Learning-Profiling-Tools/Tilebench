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
            "lnc_split": 2, "combine": "hbm_barrier", "matmul_dtype": "float32"}


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


def _prep_batch(q_t, ph_all, nrows, u_loc0, nub, tb, Ug0, D, G, DC, NBU, NUT):
    # transpose raw Q tiles into PSUM, apply phi while evacuating into ph_all
    RW = NBU * P
    pt = nl.ndarray((P, DC * RW), dtype=nl.float32, buffer=nl.psum)
    for i in range(nub):
        u = u_loc0 + i
        if G > 1:
            g = min(G, tb - u * G)
            t0 = u * G
            nisa.nc_transpose(dst=pt[0:g * D, i * P:i * P + nrows],
                              data=q_t[0:nrows, t0 * D:t0 * D + g * D],
                              engine=nisa.engine.tensor)
        else:
            for c in range(DC):
                c0 = c * P
                dc = min(P, D - c0)
                a0 = c * RW + i * P
                nisa.nc_transpose(dst=pt[0:dc, a0:a0 + nrows],
                                  data=q_t[0:nrows, u * D + c0:u * D + c0 + dc],
                                  engine=nisa.engine.tensor)
    full = 1
    if nrows != P:
        full = 0
    if G > 1:
        if (u_loc0 + nub) * G > tb:
            full = 0
    e_t = nl.ndarray((P, DC * RW), dtype=nl.float32, buffer=nl.sbuf)
    u_t = nl.ndarray((P, DC * RW), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(DC):
        cb = c * RW
        pb = c * NUT * P + Ug0 * P
        if G > 1:
            wc = G * D
        else:
            wc = min(P, D - c * P)
        if full == 1:
            ww = nub * P
            _phi(ph_all[0:wc, pb:pb + ww], pt[0:wc, cb:cb + ww],
                 e_t[0:wc, cb:cb + ww], u_t[0:wc, cb:cb + ww])
        else:
            for i in range(nub):
                if G > 1:
                    w = min(G, tb - (u_loc0 + i) * G) * D
                else:
                    w = wc
                p0 = pb + i * P
                a0 = cb + i * P
                _phi(ph_all[0:w, p0:p0 + nrows], pt[0:w, a0:a0 + nrows],
                     e_t[0:w, a0:a0 + nrows], u_t[0:w, a0:a0 + nrows])


def _q_prep_block(q_t, ph_all, nrows, tb, U0, D, G, DC, NBU, NUT):
    nu = (tb + G - 1) // G
    for s in range(0, nu, NBU):
        nub = min(NBU, nu - s)
        _prep_batch(q_t, ph_all, nrows, s, nub, tb, U0 + s, D, G, DC, NBU, NUT)


def _out_group(ph_all, o_t, nrows, u_loc0, nuo, tb, Ug0, s_mov, D, G, DC, NUT, eps):
    DA = D + 1
    op = nl.ndarray((P, BANK), dtype=nl.float32, buffer=nl.psum)
    nrt = 0
    for i in range(nuo):
        u = u_loc0 + i
        if G > 1:
            g = min(G, tb - u * G)
        else:
            g = 1
        col = i * G * DA
        for c in range(DC):
            a0 = c * NUT * P + (Ug0 + i) * P
            if G > 1:
                w = g * D
                mov = s_mov[0:w, 0:g * DA]
            else:
                w = min(P, D - c * P)
                mov = s_mov[0:w, c * DA:c * DA + DA]
            nisa.nc_matmul(dst=op[0:nrows, col:col + g * DA],
                           stationary=ph_all[0:w, a0:a0 + nrows],
                           moving=mov,
                           accumulate=(c > 0))
        nrt = nrt + g
    den = nl.ndarray((P, nrt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=den[0:nrows, 0:nrt],
                       data=op[0:nrows, D:D + (nrt - 1) * DA + 1:DA],
                       op0=nl.add, operand0=eps, engine=nisa.engine.vector)
    rden = nl.ndarray((P, nrt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rden[0:nrows, 0:nrt], data=den[0:nrows, 0:nrt])
    for r in range(nrt):
        trow = u_loc0 * G + r
        nisa.tensor_scalar(dst=o_t[0:nrows, trow * D:trow * D + D],
                           data=op[0:nrows, r * DA:r * DA + D],
                           op0=nl.multiply, operand0=rden[0:nrows, r:r + 1],
                           engine=nisa.engine.vector)


@nki.jit
def _lsa_kernel(Q, K, V, eps, G, DC, NBU, UPO, TB):
    M = Q.shape[0]
    D = Q.shape[1]
    DA = D + 1
    W = DC * DA
    SP = min(D, P)
    out = nl.ndarray((M, D), dtype=nl.float32, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R = M // n_prog
    row0 = pid * R
    T_full = R // P
    rem = R - T_full * P
    nblk = (T_full + TB - 1) // TB
    UPB = TB // G
    tb_last = T_full - (nblk - 1) * TB
    nu_last = (tb_last + G - 1) // G
    NU_main = (nblk - 1) * UPB + nu_last
    NUT = NU_main
    if rem > 0:
        NUT = NU_main + 1

    ones = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)

    ph_all = nl.ndarray((P, DC * NUT * P), dtype=nl.float32, buffer=nl.sbuf)
    s_ps = nl.ndarray((P, DC * BANK), dtype=nl.float32, buffer=nl.psum)
    scr = nl.ndarray((2 * SP, W), dtype=nl.float32, buffer=nl.shared_hbm)

    # ===== phase A: partial [S | Z] = phi(K)^T [V | 1]; phi(Q)^T prepared (lagged) =====
    q_prev = ones
    tb_prev = 1
    for b in range(nblk):
        tb = min(TB, T_full - b * TB)
        base = row0 + b * TB * P
        n = tb * D
        k_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_t, src=K.ap(pattern=[[n, P], [1, n]], offset=base * D))
        v_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=v_t, src=V.ap(pattern=[[n, P], [1, n]], offset=base * D))
        q_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_t, src=Q.ap(pattern=[[n, P], [1, n]], offset=base * D))
        _state_rows(s_ps, k_t, v_t, ones, P, tb, D, DC, b * TB)
        if b > 0:
            _q_prep_block(q_prev, ph_all, P, tb_prev, (b - 1) * UPB, D, G, DC, NBU, NUT)
        q_prev = q_t
        tb_prev = tb

    base_r = row0 + T_full * P
    k_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    v_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    q_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    if rem > 0:
        nisa.dma_copy(dst=k_r[0:rem, :], src=K[base_r:base_r + rem, :])
        nisa.dma_copy(dst=v_r[0:rem, :], src=V[base_r:base_r + rem, :])
        nisa.dma_copy(dst=q_r[0:rem, :], src=Q[base_r:base_r + rem, :])
        _state_rows(s_ps, k_r, v_r, ones, rem, 1, D, DC, T_full)

    # ---- partial state -> SBUF -> shared HBM scratch, then barrier ----
    s_own = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=s_own, value=0.0)
    for c in range(DC):
        dc = min(P, D - c * P)
        nisa.tensor_copy(dst=s_own[0:dc, c * DA:c * DA + DA],
                         src=s_ps[0:dc, c * BANK:c * BANK + DA],
                         engine=nisa.engine.vector)
    if n_prog > 1:
        nisa.dma_copy(dst=scr[pid * SP:pid * SP + SP, :], src=s_own[0:SP, :])
        nisa.core_barrier(data=scr, cores=(0, 1))

    # deferred Q preparation (overlaps the exchange latency)
    _q_prep_block(q_prev, ph_all, P, tb_prev, (nblk - 1) * UPB, D, G, DC, NBU, NUT)
    if rem > 0:
        _q_prep_block(q_r, ph_all, rem, 1, NU_main, D, G, DC, NBU, NUT)

    # ---- complete reduction over all M rows: S_tot = part0 + part1 (fixed order) ----
    if G > 1:
        s_mov = nl.ndarray((P, G * DA), dtype=nl.float32, buffer=nl.sbuf)
        if n_prog > 1:
            a_t = nl.ndarray((P, G * DA), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=a_t, value=0.0)
            b_t = nl.ndarray((P, G * DA), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=b_t, value=0.0)
            for j in range(G):
                nisa.dma_copy(dst=a_t[j * D:j * D + D, j * DA:j * DA + DA],
                              src=scr[0:D, 0:DA])
                nisa.dma_copy(dst=b_t[j * D:j * D + D, j * DA:j * DA + DA],
                              src=scr[D:2 * D, 0:DA])
            nisa.tensor_tensor(dst=s_mov, data1=a_t, data2=b_t, op=nl.add,
                               engine=nisa.engine.vector)
        else:
            nisa.memset(dst=s_mov, value=0.0)
            for j in range(G):
                nisa.dma_copy(dst=s_mov[j * D:j * D + D, j * DA:j * DA + DA],
                              src=s_own[0:D, 0:DA])
    else:
        if n_prog > 1:
            a_t = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
            b_t = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_t[0:SP, :], src=scr[0:SP, :])
            nisa.dma_copy(dst=b_t[0:SP, :], src=scr[SP:2 * SP, :])
            s_mov = nl.ndarray((P, W), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s_mov[0:SP, :], data1=a_t[0:SP, :],
                               data2=b_t[0:SP, :], op=nl.add,
                               engine=nisa.engine.vector)
        else:
            s_mov = s_own

    # ===== phase C: O = (phi(Q) S) / (phi(Q).Z + eps) =====
    for b in range(nblk):
        tb = min(TB, T_full - b * TB)
        base = row0 + b * TB * P
        n = tb * D
        o_t = nl.ndarray((P, n), dtype=nl.float32, buffer=nl.sbuf)
        nu = (tb + G - 1) // G
        for s in range(0, nu, UPO):
            nuo = min(UPO, nu - s)
            _out_group(ph_all, o_t, P, s, nuo, tb, b * UPB + s, s_mov,
                       D, G, DC, NUT, eps)
        nisa.dma_copy(dst=out.ap(pattern=[[n, P], [1, n]], offset=base * D), src=o_t)
    if rem > 0:
        o_r = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
        _out_group(ph_all, o_r, rem, 0, 1, 1, NU_main, s_mov, D, G, DC, NUT, eps)
        nisa.dma_copy(dst=out[base_r:base_r + rem, :], src=o_r[0:rem, :])

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
