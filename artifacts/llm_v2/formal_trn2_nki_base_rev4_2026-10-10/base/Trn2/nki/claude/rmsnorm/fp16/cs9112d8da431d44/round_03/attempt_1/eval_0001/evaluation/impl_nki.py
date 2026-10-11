"""RMSNorm (fp16) for AWS Trainium2, NKI 0.7.0b1, LNC=2.

y[r, k] = x[r, k] * rsqrt(mean_k(x[r, :]^2) + eps) * w[k]

Mapping (per physical core = per program, rows split in halves):
  * partition p of program pid owns rows pid*rpp*128 + p*rpp + s (s = 0..rpp-1);
    each row slot s is one "unit" held in its own SBUF tile [128, K].
  * all unit loads are issued up front (chunked <= 2048 columns); ScalarE
    computes the fp32 sum of squares per chunk (fused activation reduce),
    combines the chunk sums and evaluates -0.5*(ss/K+eps) in one instruction,
    then rsqrt; VectorE refines rstd with one Newton step and computes
    y = (x * rstd) * w in place (fp32 internally, single downcast); stores
    follow per chunk.  Every row is read from HBM exactly once.
  * w is read once into partition 0 and broadcast to all partitions with a
    K=1 matmul (exact in fp32 PSUM), copied back to fp16 (exact).
"""
import torch
import torch_neuronx  # noqa: F401  (registers the neuron device / dispatch)
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128            # partitions
_N_PROG = 2         # LNC=2: two physical cores
_NBUF = 8           # unit tiles per core (rotated if more row slots exist)
_CHUNK_CAP = 2048   # max columns per DMA / ACT / DVE instruction
_PSUM_CHUNK = 1024  # columns per PSUM tile in the weight broadcast
_MM_N = 512         # max matmul moving free size

_LAST_CONFIG = {}


def _pick(i, t0, t1, t2, t3, t4, t5, t6, t7):
    r = t7
    if i == 0:
        r = t0
    elif i == 1:
        r = t1
    elif i == 2:
        r = t2
    elif i == 3:
        r = t3
    elif i == 4:
        r = t4
    elif i == 5:
        r = t5
    elif i == 6:
        r = t6
    return r


def _load_unit(xt, xv, pid, u, K, nch, cu):
    for c in range(nch):
        c0 = c * cu
        c1 = min(K, c0 + cu)
        nisa.dma_copy(dst=xt[:, c0:c1], src=xv[pid, :, u * K + c0:u * K + c1])


def _compute_unit(xt, wb, ja, jb, hbt, ov, pid, u, K, nch, cu, hscale):
    # ---- stage 1: per-row sum of squares in fp32 (chunk partial sums) --------
    ssp = nl.ndarray((_P, nch), dtype=nl.float32, buffer=nl.sbuf)
    for c in range(nch):
        c0 = c * cu
        c1 = min(K, c0 + cu)
        jk = ja
        if c % 2 == 1:
            jk = jb
        nisa.activation(dst=jk[:, 0:c1 - c0], op=nl.square, data=xt[:, c0:c1],
                        reduce_op=nl.add, reduce_res=ssp[:, c:c + 1],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
    # h = sum_c (ssp_c * (-0.5/K) + (-0.5*eps/nch)) = -0.5 * (ss/K + eps)
    hs = nl.ndarray((_P, nch), dtype=nl.float32, buffer=nl.sbuf)
    h = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=hs, op=nl.copy, data=ssp, bias=hbt, scale=hscale,
                    reduce_op=nl.add, reduce_res=h,
                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
    # y0 ~ rsqrt(ss/K + eps)
    y0 = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=y0, op=nl.rsqrt, data=h, scale=-2.0)
    # one Newton step: rstd = y0 * (1.5 - 0.5*a*y0^2) = ((y0*h)*y0 + 1.5) * y0
    t = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=t, data=y0, op0=nl.multiply, operand0=h,
                              op1=nl.multiply, operand1=y0)
    rstd = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=rstd, data=t, op0=nl.add, operand0=1.5,
                              op1=nl.multiply, operand1=y0)

    # ---- stage 2: y = (x * rstd) * w in fp32, single downcast, in place ------
    for c in range(nch):
        c0 = c * cu
        c1 = min(K, c0 + cu)
        seg = xt[:, c0:c1]
        nisa.scalar_tensor_tensor(dst=seg, data=seg, op0=nl.multiply,
                                  operand0=rstd, op1=nl.multiply,
                                  operand1=wb[:, c0:c1])
    for c in range(nch):
        c0 = c * cu
        c1 = min(K, c0 + cu)
        nisa.dma_copy(dst=ov[pid, :, u * K + c0:u * K + c1], src=xt[:, c0:c1])


@nki.jit
def _rmsnorm_kernel(x, wt, hscale, hbias, cu, nch, pchunk):
    nd = len(x.shape)
    K = x.shape[nd - 1]
    R = 1
    for d in range(nd - 1):
        R = R * x.shape[d]
    dt = x.dtype
    out = nl.ndarray(x.shape, dtype=dt, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rpp = R // (n_prog * _P)          # row slots per partition for this program
    xv = x.reshape((n_prog, _P, rpp * K))
    ov = out.reshape((n_prog, _P, rpp * K))

    # weight row -> partition 0 of the broadcast tile (one small DMA, first)
    wb = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb[0:1, :], src=wt.reshape((1, K)))

    # independent unit tiles; issue all first loads up front
    b0 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    b1 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    b2 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    b3 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    b4 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    b5 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    b6 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    b7 = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    nfirst = min(rpp, _NBUF)
    for u in range(nfirst):
        xt0 = _pick(u, b0, b1, b2, b3, b4, b5, b6, b7)
        _load_unit(xt0, xv, pid, u, K, nch, cu)

    # broadcast w to all partitions: ones[1,128].T @ w[1, n] (exact in fp32 PSUM)
    ones = nl.ndarray((1, _P), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ps0 = nl.ndarray((_P, pchunk), dtype=nl.float32, buffer=nl.psum)
    ps1 = nl.ndarray((_P, pchunk), dtype=nl.float32, buffer=nl.psum)
    npc = (K + pchunk - 1) // pchunk
    for c in range(npc):
        c0 = c * pchunk
        c1 = min(K, c0 + pchunk)
        cw = c1 - c0
        ps = ps0
        if c % 2 == 1:
            ps = ps1
        nsub = (cw + _MM_N - 1) // _MM_N
        for n in range(nsub):
            n0 = n * _MM_N
            n1 = min(cw, n0 + _MM_N)
            nisa.nc_matmul(dst=ps[:, n0:n1], stationary=ones,
                           moving=wb[0:1, c0 + n0:c0 + n1], accumulate=False)
        nisa.tensor_copy(dst=wb[:, c0:c1], src=ps[:, 0:cw], engine=nisa.engine.vector)

    # scratch outputs of the squaring activation (only the fused sums are used)
    ja = nl.ndarray((_P, cu), dtype=nl.bfloat16, buffer=nl.sbuf)
    jb = nl.ndarray((_P, cu), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbt = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=hbt, value=hbias)

    for u in range(rpp):
        xt = _pick(u % _NBUF, b0, b1, b2, b3, b4, b5, b6, b7)
        _compute_unit(xt, wb, ja, jb, hbt, ov, pid, u, K, nch, cu, hscale)
        if u + _NBUF < rpp:
            _load_unit(xt, xv, pid, u + _NBUF, K, nch, cu)
    return out


def _config_for(K, R):
    unit = 512
    units = (K + unit - 1) // unit
    cap_units = _CHUNK_CAP // unit
    nch = (units + cap_units - 1) // cap_units
    if nch < 1:
        nch = 1
    per = (units + nch - 1) // nch
    cu = per * unit
    if cu > K:
        cu = K
    nch = (K + cu - 1) // cu
    rpp = R // (_N_PROG * _P)
    return {
        "n_programs": _N_PROG,
        "row_slots_per_partition": int(rpp),
        "unit_buffers": int(min(rpp, _NBUF)),
        "chunk_cols": int(cu),
        "n_chunks": int(nch),
        "psum_chunk": _PSUM_CHUNK,
        "rstd": "act_rsqrt+1newton",
        "weight_broadcast": "matmul_ones",
    }


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    global _LAST_CONFIG
    K = int(x.shape[-1])
    R = 1
    for d in x.shape[:-1]:
        R *= int(d)
    if R % (_N_PROG * _P) != 0:
        raise ValueError("rows (batch*M) must be a multiple of %d" % (_N_PROG * _P))
    cfg = _config_for(K, R)
    nch = cfg["n_chunks"]
    hscale = float(-0.5 / K)
    hbias = float(-0.5 * float(eps) / nch)
    out = _rmsnorm_kernel[_N_PROG](x, rms_w, hscale, hbias,
                                   cfg["chunk_cols"], nch, cfg["psum_chunk"])
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
