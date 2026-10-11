import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
BANK = 512
RES_BUDGET_BYTES = 122880      # per-partition SBUF bytes for resident input tiles
TMP_RING = 4                   # fp32 temp ring depth for the apply stage
STREAM_RING = 4                # ring depth for non-resident (streamed) tiles
APPLY_PERIOD = 3               # every APPLY_PERIOD-th tile: VectorE-only apply

_LAST_CONFIG = {}


def _stats_tile(G, S, ones16, tile, coff, np_, NB, acc):
    # per 128-channel block b:
    #   G_b = X_b^T X_b  (diag = per-channel sum of squares, fp32 accumulation)
    #   S[:, b] = X_b^T 1 (per-channel sum)
    for b in range(NB):
        c0 = coff + b * P
        nisa.nc_matmul(dst=G[:, b * P:(b + 1) * P],
                       stationary=tile[0:np_, c0:c0 + P],
                       moving=tile[0:np_, c0:c0 + P],
                       accumulate=acc)
        nisa.nc_matmul(dst=S[:, b:b + 1],
                       stationary=tile[0:np_, c0:c0 + P],
                       moving=ones16[0:np_, :],
                       accumulate=acc)


def _apply_tile(tile, coff, np_, C, SBs, SBt, tring, toff, vec_only):
    # y = x * scale + shift (fp32 math), written in place as fp16
    nisa.tensor_tensor(dst=tring[0:np_, toff:toff + C],
                       data1=tile[0:np_, coff:coff + C],
                       data2=SBs[0:np_, :], op=nl.multiply,
                       engine=nisa.engine.vector)
    if vec_only:
        nisa.tensor_tensor(dst=tile[0:np_, coff:coff + C],
                           data1=tring[0:np_, toff:toff + C],
                           data2=SBt[0:np_, :], op=nl.add,
                           engine=nisa.engine.vector)
    else:
        nisa.tensor_tensor(dst=tring[0:np_, toff:toff + C],
                           data1=tring[0:np_, toff:toff + C],
                           data2=SBt[0:np_, :], op=nl.add,
                           engine=nisa.engine.gpsimd)
        nisa.tensor_copy(dst=tile[0:np_, coff:coff + C],
                         src=tring[0:np_, toff:toff + C],
                         engine=nisa.engine.scalar)


@nki.jit
def _bn_kernel(x, gamma, beta, eps, TMAX, PERIOD):
    Nr = x.shape[0]
    C = x.shape[1]
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

    # ---------------- constants / parameters ----------------
    gbm = nl.ndarray((NB, P), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gbm, src=gamma.reshape((NB, P)))
    bbm = nl.ndarray((NB, P), dtype=beta.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=bbm, src=beta.reshape((NB, P)))

    a_i = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=a_i, pattern=[[1, P]], offset=0, channel_multiplier=1)   # p + j
    b_i = nl.ndarray((P, P), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=b_i, pattern=[[2, P]], offset=0, channel_multiplier=0)   # 2j
    a_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=a_f, src=a_i)
    b_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b_f, src=b_i)
    ident = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ident, data1=a_f, data2=b_f, op=nl.equal)       # j == p
    ident8 = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    for b in range(NB):
        nisa.tensor_copy(dst=ident8[:, b * P:(b + 1) * P], src=ident)
    ident_g = nl.ndarray((NB, NB), dtype=gamma.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=ident_g, src=ident[0:NB, 0:NB])
    ones16 = nl.ndarray((P, 1), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=ones16, value=1.0)
    ones_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)

    # gamma / beta -> block layout [p, b] = v[b*128 + p]  (exact: v * 1.0)
    GBp = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=GBp[:, 0:NB], stationary=gbm, moving=ident_g, accumulate=False)
    nisa.nc_matmul(dst=GBp[:, NB:2 * NB], stationary=bbm, moving=ident_g, accumulate=False)
    gbt = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=gbt, src=GBp)

    # ---------------- stage 2: partial statistics ----------------
    PA = nl.ndarray((P, C), dtype=f32, buffer=nl.psum)     # Gram blocks, later scale rows
    PB = nl.ndarray((P, C), dtype=f32, buffer=nl.psum)     # shift rows
    S = nl.ndarray((P, NB), dtype=f32, buffer=nl.psum)     # per-channel sums

    xres = nl.ndarray((P, TR * C), dtype=x.dtype, buffer=nl.sbuf)
    nsr = max(1, min(STREAM_RING, TS))
    sring = nl.ndarray((P, nsr * C), dtype=x.dtype, buffer=nl.sbuf)
    tring = nl.ndarray((P, TMP_RING * C), dtype=f32, buffer=nl.sbuf)

    for t in range(T):
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        acc = t > 0
        if t < TR:
            nisa.dma_copy(dst=xres[0:np_, t * C:(t + 1) * C], src=x[r0:r0 + np_, :])
            _stats_tile(PA, S, ones16, xres, t * C, np_, NB, acc)
        else:
            so = ((t - TR) % nsr) * C
            nisa.dma_copy(dst=sring[0:np_, so:so + C], src=x[r0:r0 + np_, :])
            _stats_tile(PA, S, ones16, sring, so, np_, NB, acc)

    # extract Gram diagonals -> per-channel sum of squares
    Dg = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nh = (C + BANK - 1) // BANK
    for h in range(nh):
        c0 = h * BANK
        c1 = min(C, c0 + BANK)
        nisa.tensor_tensor(dst=Dg[:, c0:c1], data1=PA[:, c0:c1], data2=ident8[:, c0:c1],
                           op=nl.multiply)
    PT = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=PT[:, 0:NB], src=S)
    nisa.tensor_reduce(dst=PT[:, NB:2 * NB], op=nl.add, data=Dg.reshape((P, NB, P)), axis=2)
    nisa.dma_copy(dst=part[pid * P:(pid + 1) * P, :], src=PT)

    if n_prog > 1:
        nisa.core_barrier(data=part, cores=(0, 1))

    # ---------------- stage 3: combine (fixed program order) ----------------
    tot = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tot, src=part[0:P, :])
    for q in range(1, n_prog):
        oth = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
        nisa.dma_copy(dst=oth, src=part[q * P:(q + 1) * P, :])
        nisa.tensor_tensor(dst=tot, data1=tot, data2=oth, op=nl.add)

    inv_n = 1.0 / float(Nr)
    mom = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=mom, data=tot, op0=nl.multiply, operand0=inv_n)
    msq = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=msq, data1=mom[:, 0:NB], data2=mom[:, 0:NB], op=nl.multiply)
    ve = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ve, data1=mom[:, NB:2 * NB], data2=msq, op=nl.subtract)
    nisa.tensor_scalar(dst=ve, data=ve, op0=nl.maximum, operand0=0.0,
                       op1=nl.add, operand1=eps)
    sq = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.sqrt, data=ve)
    rinv = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rinv, data=sq)
    tq = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    for it in range(2):
        # Newton refinement of rsqrt: r = r * (1.5 - 0.5 * v * r^2)
        nisa.tensor_tensor(dst=tq, data1=rinv, data2=rinv, op=nl.multiply)
        nisa.tensor_tensor(dst=tq, data1=tq, data2=ve, op=nl.multiply)
        nisa.tensor_scalar(dst=tq, data=tq, op0=nl.multiply, operand0=-0.5,
                           op1=nl.add, operand1=1.5)
        nisa.tensor_tensor(dst=rinv, data1=rinv, data2=tq, op=nl.multiply)

    SS = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=SS[:, 0:NB], data1=rinv, data2=gbt[:, 0:NB], op=nl.multiply)
    ms = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ms, data1=mom[:, 0:NB], data2=SS[:, 0:NB], op=nl.multiply)
    nisa.tensor_tensor(dst=SS[:, NB:2 * NB], data1=gbt[:, NB:2 * NB], data2=ms,
                       op=nl.subtract)

    # block layout -> rows broadcast to all partitions: ones^T @ diag(v_b)
    Dcat = nl.ndarray((P, 2 * C), dtype=f32, buffer=nl.sbuf)
    for t in range(2 * NB):
        if t % 2 == 0:
            nisa.tensor_scalar(dst=Dcat[:, t * P:(t + 1) * P], data=ident,
                               op0=nl.multiply, operand0=SS[:, t:t + 1],
                               engine=nisa.engine.vector)
        else:
            nisa.activation(dst=Dcat[:, t * P:(t + 1) * P], op=nl.copy, data=ident,
                            scale=SS[:, t:t + 1])
    for h in range(nh):
        c0 = h * BANK
        c1 = min(C, c0 + BANK)
        nisa.nc_matmul(dst=PA[:, c0:c1], stationary=ones_f, moving=Dcat[:, c0:c1],
                       accumulate=False)
    for h in range(nh):
        c0 = h * BANK
        c1 = min(C, c0 + BANK)
        nisa.nc_matmul(dst=PB[:, c0:c1], stationary=ones_f, moving=Dcat[:, C + c0:C + c1],
                       accumulate=False)
    SBs = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=SBs, src=PA, engine=nisa.engine.vector)
    SBt = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=SBt, src=PB, engine=nisa.engine.scalar)

    # ---------------- stage 4: apply ----------------
    for t in range(TR):
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        vec_only = ((t % PERIOD) == 0) or (np_ < P)
        _apply_tile(xres, t * C, np_, C, SBs, SBt, tring, (t % TMP_RING) * C, vec_only)
        nisa.dma_copy(dst=out[r0:r0 + np_, :], src=xres[0:np_, t * C:(t + 1) * C])
    for t in range(TR, T):
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        so = ((t - TR) % nsr) * C
        nisa.dma_copy(dst=sring[0:np_, so:so + C], src=x[r0:r0 + np_, :])
        vec_only = ((t % PERIOD) == 0) or (np_ < P)
        _apply_tile(sring, so, np_, C, SBs, SBt, tring, (t % TMP_RING) * C, vec_only)
        nisa.dma_copy(dst=out[r0:r0 + np_, :], src=sring[0:np_, so:so + C])

    return out


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    global _LAST_CONFIG
    x = input.contiguous()
    Ni = int(x.shape[0])
    Ci = int(x.shape[1])
    esize = int(x.element_size())
    tmax = max(1, RES_BUDGET_BYTES // (Ci * esize))
    out = _bn_kernel(x, gamma, beta, float(eps), tmax, APPLY_PERIOD)
    rows_per_prog = Ni // 2
    tiles = (rows_per_prog + P - 1) // P
    _LAST_CONFIG = {
        "programs": 2,
        "rows_per_program": rows_per_prog,
        "row_tile": P,
        "tiles_per_program": tiles,
        "resident_tiles": min(tiles, tmax),
        "streamed_tiles": tiles - min(tiles, tmax),
        "tmp_ring": TMP_RING,
        "stream_ring": STREAM_RING,
        "apply_period_vector_only": APPLY_PERIOD,
        "stats": "tensorE_gram_diag_plus_ones_fp32acc",
        "combine": "barrier_fixed_order_block_layout_newton_rsqrt",
        "broadcast": "tensorE_ones_x_diag_fp32",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
