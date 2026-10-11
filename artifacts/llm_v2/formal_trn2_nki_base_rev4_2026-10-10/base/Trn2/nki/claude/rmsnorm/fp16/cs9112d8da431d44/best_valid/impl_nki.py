"""RMSNorm (fp16) for AWS Trainium2, NKI 0.7.0b1, LNC=2.

y[r, k] = x[r, k] * rsqrt(mean_k(x[r, :]^2) + eps) * w[k]

Mapping (per physical core = per program, rows split in halves):
  * partition p of program pid owns the rpp consecutive rows
    pid*rpp*128 + p*rpp + s (s = 0..rpp-1).  The row slots are grouped into
    five independent SBUF tiles of sizes (1, b, b, b, 1) (rpp=8 -> 1,2,2,2,1),
    so the pipeline fills and drains with one-slot groups.
  * every row is read from HBM exactly once and kept on chip (in-place output).
  * ScalarE (only): fp32 sum of squares per row (fused activation reduce),
    chunk partials combined with a fused copy-reduce, rsqrt estimate.
    All ScalarE work for all groups is issued first (no cross-engine stalls).
  * VectorE: h = -0.5*(ss/K+eps), one Newton step for rstd, then
    y = (x * rstd) * w in one scalar_tensor_tensor (fp32 internally, a single
    downcast), stores per chunk.
  * w is read once into partition 0 and broadcast to all 128 partitions with a
    K=1 matmul (exact in fp32 PSUM), copied back to fp16 (exact).
"""
import torch
import torch_neuronx  # noqa: F401  (registers the neuron device / dispatch)
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128            # partitions
_N_PROG = 2         # LNC=2: two physical cores
_CHUNK_CAP = 5120   # max columns per DMA / ACT / DVE instruction
_PSUM_CHUNK = 1024  # columns per PSUM tile in the weight broadcast
_MM_N = 512         # max matmul moving free size

_LAST_CONFIG = {}


def _load_group(xt, xv, pid, s0, g, K, nch, cu):
    # xt: [P, g*K]; row slot j of this group sits at columns j*K..(j+1)*K
    for j in range(g):
        s = s0 + j
        for c in range(nch):
            c0 = c * cu
            c1 = min(K, c0 + cu)
            nisa.dma_copy(dst=xt[:, j * K + c0:j * K + c1],
                          src=xv[pid, :, s * K + c0:s * K + c1])


def _bcast_weight(wb, ones, ps0, ps1, K, pchunk):
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


def _stage1(xt, sa, y0, junk, junk2, biast, g, K, nch, cu, inv_k):
    # ScalarE only: sum of squares (fp32) and rsqrt estimate for g row slots
    if nch == 1:
        for j in range(g):
            nisa.activation(dst=junk[:, 0:K], op=nl.square,
                            data=xt[:, j * K:(j + 1) * K],
                            reduce_op=nl.add, reduce_res=sa[:, j:j + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
        # y0 ~ rsqrt(ss/K + eps)   (biast holds eps)
        nisa.activation(dst=y0, op=nl.rsqrt, data=sa, bias=biast, scale=inv_k)
    else:
        ssp = nl.ndarray((_P, g * nch), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(g):
            for c in range(nch):
                c0 = c * cu
                c1 = min(K, c0 + cu)
                col = j * nch + c
                nisa.activation(dst=junk[:, 0:c1 - c0], op=nl.square,
                                data=xt[:, j * K + c0:j * K + c1],
                                reduce_op=nl.add, reduce_res=ssp[:, col:col + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
        # a_j = sum_c (ssp_jc / K + eps/nch) = ss_j/K + eps   (biast holds eps/nch)
        for j in range(g):
            nisa.activation(dst=junk2, op=nl.copy, data=ssp[:, j * nch:(j + 1) * nch],
                            bias=biast, scale=inv_k,
                            reduce_op=nl.add, reduce_res=sa[:, j:j + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.activation(dst=y0, op=nl.rsqrt, data=sa, scale=1.0)


def _stage2(xt, sa, y0, wb, ov, pid, s0, g, K, nch, cu, hs, hb):
    # VectorE: h = -0.5*a, Newton step, normalise + scale in place, then stores
    h = nl.ndarray((_P, g), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=h, data=sa, op0=nl.multiply, operand0=hs,
                       op1=nl.add, operand1=hb, engine=nisa.engine.vector)
    t = nl.ndarray((_P, g), dtype=nl.float32, buffer=nl.sbuf)
    if g == 1:
        nisa.scalar_tensor_tensor(dst=t, data=y0, op0=nl.multiply, operand0=h,
                                  op1=nl.multiply, operand1=y0)
    else:
        nisa.tensor_tensor(dst=t, data1=y0, data2=y0, op=nl.multiply,
                           engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=t, data1=t, data2=h, op=nl.multiply,
                           engine=nisa.engine.vector)
    # rstd = y0 * (1.5 - 0.5*a*y0^2) = (y0*y0*h + 1.5) * y0
    rstd = nl.ndarray((_P, g), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=rstd, data=t, op0=nl.add, operand0=1.5,
                              op1=nl.multiply, operand1=y0)

    for j in range(g):
        for c in range(nch):
            c0 = c * cu
            c1 = min(K, c0 + cu)
            seg = xt[:, j * K + c0:j * K + c1]
            nisa.scalar_tensor_tensor(dst=seg, data=seg, op0=nl.multiply,
                                      operand0=rstd[:, j:j + 1], op1=nl.multiply,
                                      operand1=wb[:, c0:c1])
    for j in range(g):
        s = s0 + j
        for c in range(nch):
            c0 = c * cu
            c1 = min(K, c0 + cu)
            nisa.dma_copy(dst=ov[pid, :, s * K + c0:s * K + c1],
                          src=xt[:, j * K + c0:j * K + c1])


@nki.jit
def _rmsnorm_kernel(x, wt, inv_k, hs, hb, bval, cu, nch, pchunk, g0, g1, g2, g3, g4):
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

    s1 = g0
    s2 = s1 + g1
    s3 = s2 + g2
    s4 = s3 + g3
    assert s4 + g4 == rpp, "group sizes must cover all row slots"

    # weight row -> partition 0 of the broadcast tile (one small DMA, first)
    wb = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb[0:1, :], src=wt.reshape((1, K)))

    # independent group tiles; issue all row loads up front
    xa = nl.ndarray((_P, g0 * K), dtype=dt, buffer=nl.sbuf)
    xb = nl.ndarray((_P, g1 * K), dtype=dt, buffer=nl.sbuf)
    xc = nl.ndarray((_P, g2 * K), dtype=dt, buffer=nl.sbuf)
    xd = nl.ndarray((_P, g3 * K), dtype=dt, buffer=nl.sbuf)
    xe = nl.ndarray((_P, g4 * K), dtype=dt, buffer=nl.sbuf)
    _load_group(xa, xv, pid, 0, g0, K, nch, cu)
    _load_group(xb, xv, pid, s1, g1, K, nch, cu)
    _load_group(xc, xv, pid, s2, g2, K, nch, cu)
    _load_group(xd, xv, pid, s3, g3, K, nch, cu)
    _load_group(xe, xv, pid, s4, g4, K, nch, cu)

    # broadcast w to all partitions: ones[1,128].T @ w[1, n] (exact in fp32 PSUM)
    ones = nl.ndarray((1, _P), dtype=dt, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0)
    ps0 = nl.ndarray((_P, pchunk), dtype=nl.float32, buffer=nl.psum)
    ps1 = nl.ndarray((_P, pchunk), dtype=nl.float32, buffer=nl.psum)
    _bcast_weight(wb, ones, ps0, ps1, K, pchunk)

    # scratch outputs of the squaring / combining activations
    junk = nl.ndarray((_P, cu), dtype=nl.bfloat16, buffer=nl.sbuf)
    junk2 = nl.ndarray((_P, nch), dtype=nl.float32, buffer=nl.sbuf)
    biast = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=biast, value=bval)

    # per-group row statistics (separate tiles per group)
    sa_a = nl.ndarray((_P, g0), dtype=nl.float32, buffer=nl.sbuf)
    sa_b = nl.ndarray((_P, g1), dtype=nl.float32, buffer=nl.sbuf)
    sa_c = nl.ndarray((_P, g2), dtype=nl.float32, buffer=nl.sbuf)
    sa_d = nl.ndarray((_P, g3), dtype=nl.float32, buffer=nl.sbuf)
    sa_e = nl.ndarray((_P, g4), dtype=nl.float32, buffer=nl.sbuf)
    y_a = nl.ndarray((_P, g0), dtype=nl.float32, buffer=nl.sbuf)
    y_b = nl.ndarray((_P, g1), dtype=nl.float32, buffer=nl.sbuf)
    y_c = nl.ndarray((_P, g2), dtype=nl.float32, buffer=nl.sbuf)
    y_d = nl.ndarray((_P, g3), dtype=nl.float32, buffer=nl.sbuf)
    y_e = nl.ndarray((_P, g4), dtype=nl.float32, buffer=nl.sbuf)

    # stage 1 (ScalarE) for every group first
    _stage1(xa, sa_a, y_a, junk, junk2, biast, g0, K, nch, cu, inv_k)
    _stage1(xb, sa_b, y_b, junk, junk2, biast, g1, K, nch, cu, inv_k)
    _stage1(xc, sa_c, y_c, junk, junk2, biast, g2, K, nch, cu, inv_k)
    _stage1(xd, sa_d, y_d, junk, junk2, biast, g3, K, nch, cu, inv_k)
    _stage1(xe, sa_e, y_e, junk, junk2, biast, g4, K, nch, cu, inv_k)

    # stage 2 (VectorE) + stores, group by group
    _stage2(xa, sa_a, y_a, wb, ov, pid, 0, g0, K, nch, cu, hs, hb)
    _stage2(xb, sa_b, y_b, wb, ov, pid, s1, g1, K, nch, cu, hs, hb)
    _stage2(xc, sa_c, y_c, wb, ov, pid, s2, g2, K, nch, cu, hs, hb)
    _stage2(xd, sa_d, y_d, wb, ov, pid, s3, g3, K, nch, cu, hs, hb)
    _stage2(xe, sa_e, y_e, wb, ov, pid, s4, g4, K, nch, cu, hs, hb)
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
    mid = rpp - 2
    b = mid // 3
    r = mid % 3
    ga = b + (1 if r > 0 else 0)
    gb = b + (1 if r > 1 else 0)
    groups = [1, ga, gb, b, 1]
    return {
        "n_programs": _N_PROG,
        "row_slots_per_partition": int(rpp),
        "groups": [int(v) for v in groups],
        "chunk_cols": int(cu),
        "n_chunks": int(nch),
        "psum_chunk": _PSUM_CHUNK,
        "rstd": "act_rsqrt+1newton_dve",
        "weight_broadcast": "matmul_ones",
    }


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    global _LAST_CONFIG
    K = int(x.shape[-1])
    R = 1
    for d in x.shape[:-1]:
        R *= int(d)
    if R % (_N_PROG * _P) != 0 or R // (_N_PROG * _P) < 5:
        raise ValueError("rows (batch*M) must be a multiple of %d and >= %d"
                         % (_N_PROG * _P, 5 * _N_PROG * _P))
    cfg = _config_for(K, R)
    nch = cfg["n_chunks"]
    inv_k = float(1.0 / K)
    if nch == 1:
        hs = float(-0.5 / K)
        hb = float(-0.5 * float(eps))
        bval = float(eps)
    else:
        hs = -0.5
        hb = 0.0
        bval = float(eps) / nch
    g = cfg["groups"]
    out = _rmsnorm_kernel[_N_PROG](x, rms_w, inv_k, hs, hb, bval,
                                   cfg["chunk_cols"], nch, cfg["psum_chunk"],
                                   g[0], g[1], g[2], g[3], g[4])
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
