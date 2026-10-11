import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
BANK = 512
RES_BUDGET_BYTES = 143360      # per-partition SBUF bytes for resident input tiles (70 tiles @ C=1024 fp16)
TMP_RING = 2                   # fp32 temp ring for the apply stage
STREAM_RING = 3                # ring depth for non-resident (streamed) tiles

_LAST_CONFIG = {}


def _gram_dst(G0, G1, col):
    if col < BANK:
        return G0[:, col:col + P]
    return G1[:, col - BANK:col - BANK + P]


def _stats_tile(G0, G1, Ssum, ones16, src, coff, np_, NB, acc):
    # per 128-channel block b:
    #   G_b = X_b^T X_b   (diag = per-channel sum of squares, exact fp16 products, fp32 accumulation)
    #   Ssum[:, b] = X_b^T 1  (per-channel sum)
    for b in range(NB):
        c0 = coff + b * P
        nisa.nc_matmul(dst=_gram_dst(G0, G1, b * P),
                       stationary=src[0:np_, c0:c0 + P],
                       moving=src[0:np_, c0:c0 + P],
                       accumulate=acc)
        nisa.nc_matmul(dst=Ssum[:, b:b + 1],
                       stationary=src[0:np_, c0:c0 + P],
                       moving=ones16[0:np_, :],
                       accumulate=acc)


def _apply_tile(src, coff, np_, C, SBs, SBt, tring, toff):
    # y = x * scale + shift in fp32, stored in place as the input dtype
    nisa.tensor_tensor(dst=tring[0:np_, toff:toff + C],
                       data1=src[0:np_, coff:coff + C],
                       data2=SBs[0:np_, :], op=nl.multiply,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=src[0:np_, coff:coff + C],
                       data1=tring[0:np_, toff:toff + C],
                       data2=SBt[0:np_, :], op=nl.add,
                       engine=nisa.engine.vector)


@nki.jit
def _bn_kernel(x, gamma, beta, eps, TMAX):
    Nr = x.shape[0]
    C = x.shape[1]
    assert C == 2 * BANK, f"kernel expects C == 1024, got {C}"
    NB = C // P
    f32 = nl.float32
    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    out = nl.ndarray((Nr, C), dtype=x.dtype, buffer=nl.shared_hbm)
    part = nl.ndarray((n_prog * P, 2 * NB), dtype=f32, buffer=nl.shared_hbm)

    r_start = (pid * Nr) // n_prog
    r_end = ((pid + 1) * Nr) // n_prog
    R = r_end - r_start
    T = (R + P - 1) // P
    TR = min(T, TMAX)
    TS = T - TR
    nsr = max(1, min(STREAM_RING, TS))

    # small parameter loads first (so they do not queue behind the big loads)
    gbm = nl.ndarray((NB, P), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gbm, src=gamma.reshape((NB, P)))
    bbm = nl.ndarray((NB, P), dtype=beta.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bbm, src=beta.reshape((NB, P)))

    ones16 = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones16, value=1.0)

    xres = nl.ndarray((P, TR * C), dtype=x.dtype, buffer=nl.sbuf)
    sring = nl.ndarray((P, nsr * C), dtype=x.dtype, buffer=nl.sbuf)

    G0 = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum)
    G1 = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum)
    Ssum = nl.ndarray((P, NB), dtype=f32, buffer=nl.psum)

    # ---------------- stage 2: partial statistics ----------------
    for t in range(T):
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        acc = t > 0
        if t < TR:
            nisa.dma_copy(dst=xres[0:np_, t * C:(t + 1) * C], src=x[r0:r0 + np_, :])
            _stats_tile(G0, G1, Ssum, ones16, xres, t * C, np_, NB, acc)
        else:
            so = ((t - TR) % nsr) * C
            nisa.dma_copy(dst=sring[0:np_, so:so + C], src=x[r0:r0 + np_, :])
            _stats_tile(G0, G1, Ssum, ones16, sring, so, np_, NB, acc)

    # ---------------- constants ----------------
    a_i = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=a_i, pattern=[[1, P]], offset=0, channel_multiplier=1)   # p + j
    b_i = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=b_i, pattern=[[2, P]], offset=0, channel_multiplier=0)   # 2j
    a_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=a_f, src=a_i, engine=nisa.engine.vector)
    b_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b_f, src=b_i, engine=nisa.engine.vector)
    ident = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ident, data1=a_f, data2=b_f, op=nl.equal,
                       engine=nisa.engine.vector)                           # j == p
    ident4 = nl.ndarray((P, BANK), dtype=f32, buffer=nl.sbuf)
    for k in range(BANK // P):
        nisa.tensor_copy(dst=ident4[:, k * P:(k + 1) * P], src=ident, engine=nisa.engine.vector)
    ident_g = nl.ndarray((NB, NB), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ident_g, src=ident[0:NB, 0:NB], engine=nisa.engine.vector)
    ones_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)

    # gamma / beta -> block layout [p, b] = v[b*128 + p] (exact: v * 1.0, fp32 accumulation)
    GBp = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=GBp[:, 0:NB], stationary=gbm, moving=ident_g, accumulate=False)
    nisa.nc_matmul(dst=GBp[:, NB:2 * NB], stationary=bbm, moving=ident_g, accumulate=False)
    gbt = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gbt, src=GBp, engine=nisa.engine.vector)

    # ---------------- per-program partials (block layout) ----------------
    Dg = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=Dg[:, 0:BANK], data1=G0, data2=ident4, op=nl.multiply,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=Dg[:, BANK:2 * BANK], data1=G1, data2=ident4, op=nl.multiply,
                       engine=nisa.engine.vector)
    PT = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=PT[:, 0:NB], src=Ssum, engine=nisa.engine.vector)
    nisa.tensor_reduce(dst=PT[:, NB:2 * NB], op=nl.add, data=Dg.reshape((P, NB, P)), axis=2)
    nisa.dma_copy(dst=part[pid * P:(pid + 1) * P, :], src=PT)

    if n_prog > 1:
        nisa.core_barrier(data=part, cores=(0, 1))

    # ---------------- stage 3: combine (part0 + part1, commutative -> identical on both cores) ----
    tot = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    if n_prog > 1:
        q = 1 - pid
        oth = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=oth, src=part[q * P:(q + 1) * P, :])
        nisa.tensor_tensor(dst=tot, data1=PT, data2=oth, op=nl.add, engine=nisa.engine.vector)
    else:
        nisa.tensor_copy(dst=tot, src=PT, engine=nisa.engine.vector)

    inv_n = 1.0 / float(Nr)
    mom = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mom, data=tot, op0=nl.multiply, operand0=inv_n,
                       engine=nisa.engine.vector)
    msq = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=msq, data1=mom[:, 0:NB], data2=mom[:, 0:NB], op=nl.multiply,
                       engine=nisa.engine.vector)
    ve = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ve, data1=mom[:, NB:2 * NB], data2=msq, op=nl.subtract,
                       engine=nisa.engine.vector)
    nisa.tensor_scalar(dst=ve, data=ve, op0=nl.maximum, operand0=0.0,
                       op1=nl.add, operand1=eps, engine=nisa.engine.vector)
    sq = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.sqrt, data=ve)
    rinv = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rinv, data=sq)
    tq = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    for it in range(2):
        # Newton refinement of rsqrt: r = r * (1.5 - 0.5 * v * r^2)
        nisa.tensor_tensor(dst=tq, data1=rinv, data2=rinv, op=nl.multiply,
                           engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=tq, data1=tq, data2=ve, op=nl.multiply,
                           engine=nisa.engine.vector)
        nisa.tensor_scalar(dst=tq, data=tq, op0=nl.multiply, operand0=-0.5,
                           op1=nl.add, operand1=1.5, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=rinv, data1=rinv, data2=tq, op=nl.multiply,
                           engine=nisa.engine.vector)

    SS = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=SS[:, 0:NB], data1=rinv, data2=gbt[:, 0:NB], op=nl.multiply,
                       engine=nisa.engine.vector)
    ms = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ms, data1=mom[:, 0:NB], data2=SS[:, 0:NB], op=nl.multiply,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=SS[:, NB:2 * NB], data1=gbt[:, NB:2 * NB], data2=ms,
                       op=nl.subtract, engine=nisa.engine.vector)

    # block layout -> rows broadcast to all partitions: ones^T @ diag(v_b)
    Dcat = nl.ndarray((P, 2 * C), dtype=f32, buffer=nl.sbuf)
    for t in range(2 * NB):
        nisa.tensor_scalar(dst=Dcat[:, t * P:(t + 1) * P], data=ident,
                           op0=nl.multiply, operand0=SS[:, t:t + 1],
                           engine=nisa.engine.vector)
    T0 = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum)
    T1 = nl.ndarray((P, BANK), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=G0, stationary=ones_f, moving=Dcat[:, 0:BANK], accumulate=False)
    nisa.nc_matmul(dst=G1, stationary=ones_f, moving=Dcat[:, BANK:2 * BANK], accumulate=False)
    nisa.nc_matmul(dst=T0, stationary=ones_f, moving=Dcat[:, C:C + BANK], accumulate=False)
    nisa.nc_matmul(dst=T1, stationary=ones_f, moving=Dcat[:, C + BANK:C + 2 * BANK],
                   accumulate=False)
    SBs = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    SBt = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=SBs[:, 0:BANK], src=G0, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=SBs[:, BANK:2 * BANK], src=G1, engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=SBt[:, 0:BANK], src=T0, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=SBt[:, BANK:2 * BANK], src=T1, engine=nisa.engine.scalar)

    # ---------------- stage 4: apply ----------------
    tring = nl.ndarray((P, TMP_RING * C), dtype=f32, buffer=nl.sbuf)
    cnt = 0
    nhold = min(nsr, TS)          # streamed tiles still held in the ring after stage 2
    nre = TS - nhold              # streamed tiles that must be reloaded

    # (a) streamed tiles still in the ring
    for s in range(nre, TS):
        t = TR + s
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        so = (s % nsr) * C
        toff = (cnt % TMP_RING) * C
        cnt = cnt + 1
        _apply_tile(sring, so, np_, C, SBs, SBt, tring, toff)
        nisa.dma_copy(dst=out[r0:r0 + np_, :], src=sring[0:np_, so:so + C])

    # prefetch the first reloads
    for k in range(min(nsr, nre)):
        t = TR + k
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        so = (k % nsr) * C
        nisa.dma_copy(dst=sring[0:np_, so:so + C], src=x[r0:r0 + np_, :])

    # (b) resident tiles
    for t in range(TR):
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        toff = (cnt % TMP_RING) * C
        cnt = cnt + 1
        _apply_tile(xres, t * C, np_, C, SBs, SBt, tring, toff)
        nisa.dma_copy(dst=out[r0:r0 + np_, :], src=xres[0:np_, t * C:(t + 1) * C])

    # (c) reloaded streamed tiles
    for k in range(nre):
        t = TR + k
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        so = (k % nsr) * C
        toff = (cnt % TMP_RING) * C
        cnt = cnt + 1
        _apply_tile(sring, so, np_, C, SBs, SBt, tring, toff)
        nisa.dma_copy(dst=out[r0:r0 + np_, :], src=sring[0:np_, so:so + C])
        if k + nsr < nre:
            t2 = TR + k + nsr
            r2 = r_start + t2 * P
            np2 = min(P, r_end - r2)
            nisa.dma_copy(dst=sring[0:np2, so:so + C], src=x[r2:r2 + np2, :])

    return out


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    global _LAST_CONFIG
    x = input.contiguous()
    Ni = int(x.shape[0])
    Ci = int(x.shape[1])
    esize = int(x.element_size())
    tmax = max(1, RES_BUDGET_BYTES // (Ci * esize))
    out = _bn_kernel(x, gamma, beta, float(eps), tmax)
    rows_per_prog = Ni // 2
    tiles = (rows_per_prog + P - 1) // P
    res = min(tiles, tmax)
    streamed = tiles - res
    _LAST_CONFIG = {
        "programs": 2,
        "rows_per_program": rows_per_prog,
        "row_tile": P,
        "tiles_per_program": tiles,
        "resident_tiles": res,
        "streamed_tiles": streamed,
        "tmp_ring": TMP_RING,
        "stream_ring": STREAM_RING,
        "apply": "vectorE_mul_add_fp32",
        "stats": "tensorE_gram_diag_plus_ones_fp32acc",
        "combine": "barrier_fixed_order_block_layout_newton_rsqrt",
        "broadcast": "tensorE_ones_x_diag_fp32_per_bank",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
