import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Two-pass training-mode batch norm (fp16 in/out, fp32 statistics), one launch under LNC=2.
#  stage 2: each program owns half the rows. Per row tile (128 rows, zero padded):
#           - per-channel sums: fp32 per-partition accumulation on VectorE (level-1, sequential),
#             finished by a ones-matmul over partitions (fp32);
#           - per-channel sums of squares: diag of X_b^T X_b on TensorE (exact fp16 products,
#             fp32 PSUM accumulation). Each Gram PSUM bank is zero-initialised by ONE full-bank
#             matmul and afterwards only accumulated into (no sub-bank accumulate=False writes).
#           Partials (block layout [p, b] = channel b*128+p) -> shared HBM scratch.
#  stage 3: core barrier, fixed-order combine part0 + part1, mean / clamped var / rsqrt
#           (sqrt + reciprocal + 2 Newton steps), scale = inv_std*gamma, shift = beta - mean*scale,
#           broadcast to all partitions via ones^T @ diag(v) on TensorE (fp32).
#  stage 4: y = x*scale + shift in fp32 on VectorE, stored as fp16 (resident tiles reused,
#           remaining tiles re-read).

_ROW_TILE = 128
_RES_BUDGET_BYTES = 143360     # per-partition SBUF bytes for resident row tiles (70 x 1024 fp16)
_STREAM_RING = 3               # ring slots for non-resident row tiles
_KERNEL_TAG = 50517            # compile-cache key component of this kernel revision
_LAST_CONFIG = {}


@nki.jit
def bn_fwd_fp16_twopass_v5(x, gamma, beta, eps, tmax, nring, ktag):
    P = 128
    BK = 512
    Nr = x.shape[0]
    C = x.shape[1]
    assert C == 2 * BK, f"kernel expects C == 1024, got {C}"
    assert ktag > 0, "invalid kernel tag"
    NB = C // P            # 8 channel blocks of 128
    BPB = BK // P          # 4 channel blocks per PSUM bank
    f32 = nl.float32
    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    out = nl.ndarray((Nr, C), dtype=x.dtype, buffer=nl.shared_hbm)
    part = nl.ndarray((n_prog * P, 2 * NB), dtype=f32, buffer=nl.shared_hbm)

    r_start = (pid * Nr) // n_prog
    r_end = ((pid + 1) * Nr) // n_prog
    T = (r_end - r_start + P - 1) // P
    TR = min(T, tmax)
    TS = T - TR
    nsr = max(1, min(nring, TS))

    # ---------------- gamma / beta -> fp32 block layout (DMA upcasts) ----------------
    gbf = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    g3 = gamma.reshape((NB, P, 1))
    b3 = beta.reshape((NB, P, 1))
    for b in range(NB):
        nisa.dma_copy(dst=gbf[:, b:b + 1], src=g3[b])
        nisa.dma_copy(dst=gbf[:, NB + b:NB + b + 1], src=b3[b])

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
    ident4 = nl.ndarray((P, BK), dtype=f32, buffer=nl.sbuf)
    for k in range(BPB):
        nisa.tensor_copy(dst=ident4[:, k * P:(k + 1) * P], src=ident, engine=nisa.engine.vector)
    ones_f = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
    nisa.memset(dst=ones_f, value=1.0)
    zeros16 = nl.ndarray((P, BK), dtype=x.dtype, buffer=nl.sbuf)
    nisa.memset(dst=zeros16, value=0.0)

    # Gram accumulators: one full-bank zeroing matmul each, then accumulate-only writes.
    Q0 = nl.ndarray((P, BK), dtype=f32, buffer=nl.psum)
    Q1 = nl.ndarray((P, BK), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=Q0, stationary=zeros16[:, 0:P], moving=zeros16, accumulate=False)
    nisa.nc_matmul(dst=Q1, stationary=zeros16[:, 0:P], moving=zeros16, accumulate=False)

    xall = nl.ndarray((P, (TR + nsr) * C), dtype=x.dtype, buffer=nl.sbuf)
    accS = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)

    # ---------------- stage 2: partial statistics ----------------
    for t in range(T):
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        if t < TR:
            slot = t
        else:
            slot = TR + ((t - TR) % nsr)
        off = slot * C
        if np_ < P:
            nisa.memset(dst=xall[:, off:off + C], value=0.0)
        nisa.dma_copy(dst=xall[0:np_, off:off + C], src=x[r0:r0 + np_, :])
        if t == 0:
            nisa.tensor_copy(dst=accS, src=xall[:, off:off + C], engine=nisa.engine.vector)
        else:
            nisa.tensor_tensor(dst=accS, data1=accS, data2=xall[:, off:off + C],
                               op=nl.add, engine=nisa.engine.vector)
        for b in range(NB):
            c0 = off + b * P
            if b < BPB:
                nisa.nc_matmul(dst=Q0[:, b * P:(b + 1) * P],
                               stationary=xall[:, c0:c0 + P],
                               moving=xall[:, c0:c0 + P],
                               accumulate=True)
            else:
                bb = b - BPB
                nisa.nc_matmul(dst=Q1[:, bb * P:(bb + 1) * P],
                               stationary=xall[:, c0:c0 + P],
                               moving=xall[:, c0:c0 + P],
                               accumulate=True)

    # per-channel sums broadcast over partitions: ones^T @ accS (full banks)
    S0 = nl.ndarray((P, BK), dtype=f32, buffer=nl.psum)
    S1 = nl.ndarray((P, BK), dtype=f32, buffer=nl.psum)
    nisa.nc_matmul(dst=S0, stationary=ones_f, moving=accS[:, 0:BK], accumulate=False)
    nisa.nc_matmul(dst=S1, stationary=ones_f, moving=accS[:, BK:C], accumulate=False)

    # diagonal extraction -> block layout PT[p, b] = sum, PT[p, NB + b] = sumsq (channel b*128+p)
    D = nl.ndarray((P, 4 * BK), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=D[:, 0:BK], data1=S0, data2=ident4, op=nl.multiply,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=D[:, BK:2 * BK], data1=S1, data2=ident4, op=nl.multiply,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=D[:, 2 * BK:3 * BK], data1=Q0, data2=ident4, op=nl.multiply,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=D[:, 3 * BK:4 * BK], data1=Q1, data2=ident4, op=nl.multiply,
                       engine=nisa.engine.vector)
    PT = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=PT, op=nl.add, data=D.reshape((P, 2 * NB, P)), axis=2)

    # partials -> scratch
    nisa.dma_copy(dst=part[pid * P:(pid + 1) * P, :], src=PT)

    # ---------------- stage 3: combine (part0 + part1; fp32 add is commutative) ----------------
    tot = nl.ndarray((P, 2 * NB), dtype=f32, buffer=nl.sbuf)
    if n_prog > 1:
        nisa.core_barrier(data=part, cores=(0, 1))
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
    nisa.tensor_tensor(dst=SS[:, 0:NB], data1=rinv, data2=gbf[:, 0:NB], op=nl.multiply,
                       engine=nisa.engine.vector)
    ms = nl.ndarray((P, NB), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ms, data1=mom[:, 0:NB], data2=SS[:, 0:NB], op=nl.multiply,
                       engine=nisa.engine.vector)
    nisa.tensor_tensor(dst=SS[:, NB:2 * NB], data1=gbf[:, NB:2 * NB], data2=ms,
                       op=nl.subtract, engine=nisa.engine.vector)

    # block layout -> per-channel rows broadcast to all partitions: ones^T @ diag(v_b)
    for t in range(2 * NB):
        nisa.tensor_scalar(dst=D[:, t * P:(t + 1) * P], data=ident,
                           op0=nl.multiply, operand0=SS[:, t:t + 1],
                           engine=nisa.engine.vector)
    nisa.nc_matmul(dst=S0, stationary=ones_f, moving=D[:, 0:BK], accumulate=False)
    nisa.nc_matmul(dst=S1, stationary=ones_f, moving=D[:, BK:2 * BK], accumulate=False)
    nisa.nc_matmul(dst=Q0, stationary=ones_f, moving=D[:, 2 * BK:3 * BK], accumulate=False)
    nisa.nc_matmul(dst=Q1, stationary=ones_f, moving=D[:, 3 * BK:4 * BK], accumulate=False)
    SBs = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    SBt = nl.ndarray((P, C), dtype=f32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=SBs[:, 0:BK], src=S0, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=SBs[:, BK:2 * BK], src=S1, engine=nisa.engine.scalar)
    nisa.tensor_copy(dst=SBt[:, 0:BK], src=Q0, engine=nisa.engine.vector)
    nisa.tensor_copy(dst=SBt[:, BK:2 * BK], src=Q1, engine=nisa.engine.scalar)

    # ---------------- stage 4: apply ----------------
    tring = nl.ndarray((P, 2 * C), dtype=f32, buffer=nl.sbuf)
    nhold = min(nsr, TS)          # streamed tiles still held in the ring
    nre = TS - nhold              # streamed tiles that must be re-read
    for i in range(T):
        if i < nhold:
            t = TR + nre + i
            slot = TR + ((nre + i) % nsr)
        elif i < nhold + TR:
            t = i - nhold
            slot = t
        else:
            t = TR + (i - nhold - TR)
            slot = TR + ((i - nhold - TR) % nsr)
        if i == nhold:
            for k in range(min(nsr, nre)):
                rk = r_start + (TR + k) * P
                nk = min(P, r_end - rk)
                ok = (TR + (k % nsr)) * C
                nisa.dma_copy(dst=xall[0:nk, ok:ok + C], src=x[rk:rk + nk, :])
        r0 = r_start + t * P
        np_ = min(P, r_end - r0)
        off = slot * C
        toff = (i % 2) * C
        nisa.tensor_tensor(dst=tring[0:np_, toff:toff + C], data1=xall[0:np_, off:off + C],
                           data2=SBs[0:np_, :], op=nl.multiply, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=xall[0:np_, off:off + C], data1=tring[0:np_, toff:toff + C],
                           data2=SBt[0:np_, :], op=nl.add, engine=nisa.engine.vector)
        nisa.dma_copy(dst=out[r0:r0 + np_, :], src=xall[0:np_, off:off + C])
        if i >= nhold + TR:
            s2 = (i - nhold - TR) + nsr
            if s2 < nre:
                r2 = r_start + (TR + s2) * P
                n2 = min(P, r_end - r2)
                o2 = (TR + (s2 % nsr)) * C
                nisa.dma_copy(dst=xall[0:n2, o2:o2 + C], src=x[r2:r2 + n2, :])

    return out


def run(input: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
        N: int, C: int, eps: float, **kwargs):
    global _LAST_CONFIG
    x = input.contiguous()
    Ni = int(x.shape[0])
    Ci = int(x.shape[1])
    esize = int(x.element_size())
    tmax = max(1, _RES_BUDGET_BYTES // (Ci * esize))
    out = bn_fwd_fp16_twopass_v5(x, gamma, beta, float(eps), int(tmax),
                                 int(_STREAM_RING), int(_KERNEL_TAG))
    rows_per_prog = Ni // 2
    tiles = (rows_per_prog + _ROW_TILE - 1) // _ROW_TILE
    res = min(tiles, tmax)
    streamed = tiles - res
    _LAST_CONFIG = {
        "programs": 2,
        "rows_per_program": rows_per_prog,
        "row_tile": _ROW_TILE,
        "tiles_per_program": tiles,
        "resident_tiles": res,
        "streamed_tiles": streamed,
        "stream_ring": _STREAM_RING,
        "stats_sum": "vectorE_fp32_partition_accum_then_ones_matmul",
        "stats_sumsq": "tensorE_gram_diag_fp32acc_fullbank_zero_init",
        "combine": "core_barrier_fixed_order_block_layout_newton_rsqrt",
        "broadcast": "tensorE_ones_x_diag_fp32_full_bank",
        "apply": "vectorE_mul_add_fp32",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
