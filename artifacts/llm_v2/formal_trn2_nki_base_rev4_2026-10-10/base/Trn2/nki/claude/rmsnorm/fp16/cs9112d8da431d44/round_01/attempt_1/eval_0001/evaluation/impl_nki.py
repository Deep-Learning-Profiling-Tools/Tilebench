"""RMSNorm (fp16) for AWS Trainium2, NKI 0.7.0b1, LNC=2.

y[r, k] = x[r, k] * rsqrt(mean_k(x[r, :]^2) + eps) * w[k]

Mapping (per physical core = per program, rows split in halves):
  * partition p of program pid owns the 8 consecutive rows pid*1024 + p*8 + s,
    s = 0..7, grouped into 4 independent SBUF tiles (2 rows / partition each).
  * every row is read from HBM exactly once and kept on chip (in place output).
  * ScalarE: sum of squares (fp32 fused activation reduce), rstd estimate.
  * VectorE: Newton refinement of rstd (tiny), normalise+scale in one
    scalar_tensor_tensor ((x * rstd) * w, fp32 internally, single downcast).
  * w is read once from HBM into one partition and broadcast to all 128
    partitions with a K=1 matmul (exact), then copied back to fp16 (exact).
"""
import torch
import torch_neuronx  # noqa: F401  (registers the neuron device / dispatch)
import nki
import nki.language as nl
import nki.isa as nisa

_P = 128            # partitions
_N_PROG = 2         # LNC=2: two physical cores
_NGROUPS = 4        # independent SBUF tiles per core
_CHUNK_CAP = 5120   # max columns per DMA / ACT / DVE instruction
_PSUM_CHUNK = 1024  # columns per PSUM tile in the weight broadcast
_MM_N = 512         # max matmul moving free size

_LAST_CONFIG = {}


def _load_group(xt, xv, pid, s0, jg, K, nch, cu):
    # xt: [P, jg*K] SBUF tile; row slot j of this group sits at columns j*K..(j+1)*K
    for j in range(jg):
        s = s0 + j
        for c in range(nch):
            c0 = c * cu
            c1 = min(K, c0 + cu)
            nisa.dma_copy(dst=xt[:, j * K + c0:j * K + c1],
                          src=xv[pid, :, s * K + c0:s * K + c1])


def _finish_group(xt, wb, junk, epst, nhet, ov, pid, s0, jg, K, nch, cu, inv_k):
    # ---- stage 1: per-row sum of squares (fp32), chunk partial sums -------------
    ssp = nl.ndarray((_P, jg * nch), dtype=nl.float32, buffer=nl.sbuf)
    for j in range(jg):
        for c in range(nch):
            c0 = c * cu
            c1 = min(K, c0 + cu)
            col = c * jg + j
            nisa.activation(dst=junk[:, 0:c1 - c0], op=nl.square,
                            data=xt[:, j * K + c0:j * K + c1],
                            reduce_op=nl.add, reduce_res=ssp[:, col:col + 1],
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
    ss = ssp
    if nch > 1:
        ss = nl.ndarray((_P, jg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ss, data1=ssp[:, 0:jg], data2=ssp[:, jg:2 * jg], op=nl.add)
        for c in range(2, nch):
            nisa.tensor_tensor(dst=ss, data1=ss, data2=ssp[:, c * jg:(c + 1) * jg], op=nl.add)

    # ---- rstd = rsqrt(ss/K + eps), one Newton step for accuracy -----------------
    # h = -0.5 * (ss/K + eps)
    h = nl.ndarray((_P, jg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=h, op=nl.copy, data=ss, bias=nhet, scale=-0.5 * inv_k)
    # y0 ~ rsqrt(ss/K + eps)
    y0 = nl.ndarray((_P, jg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=y0, op=nl.rsqrt, data=ss, bias=epst, scale=inv_k)
    # rstd = y0 * (1.5 - 0.5 * a * y0^2) = (y0*y0*h + 1.5) * y0
    t = nl.ndarray((_P, jg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=t, data1=y0, data2=y0, op=nl.multiply)
    nisa.tensor_tensor(dst=t, data1=t, data2=h, op=nl.multiply)
    rstd = nl.ndarray((_P, jg), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(dst=rstd, data=t, op0=nl.add, operand0=1.5,
                              op1=nl.multiply, operand1=y0)

    # ---- stage 2: y = (x * rstd) * w in fp32, single downcast, in place ---------
    for j in range(jg):
        for c in range(nch):
            c0 = c * cu
            c1 = min(K, c0 + cu)
            seg = xt[:, j * K + c0:j * K + c1]
            nisa.scalar_tensor_tensor(dst=seg, data=seg, op0=nl.multiply,
                                      operand0=rstd[:, j:j + 1], op1=nl.multiply,
                                      operand1=wb[:, c0:c1])
    for j in range(jg):
        s = s0 + j
        for c in range(nch):
            c0 = c * cu
            c1 = min(K, c0 + cu)
            nisa.dma_copy(dst=ov[pid, :, s * K + c0:s * K + c1],
                          src=xt[:, j * K + c0:j * K + c1])


@nki.jit
def _rmsnorm_kernel(x, wt, eps, inv_k, cu, nch, pchunk):
    nd = len(x.shape)
    K = x.shape[nd - 1]
    R = 1
    for d in range(nd - 1):
        R = R * x.shape[d]
    dt = x.dtype
    out = nl.ndarray(x.shape, dtype=dt, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rpp = R // (n_prog * _P)          # rows per partition handled by this program
    jg = rpp // _NGROUPS              # rows per partition per group tile
    xv = x.reshape((n_prog, _P, rpp * K))
    ov = out.reshape((n_prog, _P, rpp * K))

    # weight row -> partition 0 of the broadcast tile (one small DMA)
    wb = nl.ndarray((_P, K), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=wb[0:1, :], src=wt.reshape((1, K)))

    # issue all row loads first (independent tiles, everything stays on chip)
    xa = nl.ndarray((_P, jg * K), dtype=dt, buffer=nl.sbuf)
    xb = nl.ndarray((_P, jg * K), dtype=dt, buffer=nl.sbuf)
    xc = nl.ndarray((_P, jg * K), dtype=dt, buffer=nl.sbuf)
    xd = nl.ndarray((_P, jg * K), dtype=dt, buffer=nl.sbuf)
    _load_group(xa, xv, pid, 0 * jg, jg, K, nch, cu)
    _load_group(xb, xv, pid, 1 * jg, jg, K, nch, cu)
    _load_group(xc, xv, pid, 2 * jg, jg, K, nch, cu)
    _load_group(xd, xv, pid, 3 * jg, jg, K, nch, cu)

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
        if c % 2 == 0:
            ps = ps0
        else:
            ps = ps1
        nsub = (cw + _MM_N - 1) // _MM_N
        for n in range(nsub):
            n0 = n * _MM_N
            n1 = min(cw, n0 + _MM_N)
            nisa.nc_matmul(dst=ps[:, n0:n1], stationary=ones,
                           moving=wb[0:1, c0 + n0:c0 + n1], accumulate=False)
        nisa.tensor_copy(dst=wb[:, c0:c1], src=ps[:, 0:cw], engine=nisa.engine.vector)

    # scratch output of the squaring activation (only its fused reduction is used)
    junk = nl.ndarray((_P, cu), dtype=nl.bfloat16, buffer=nl.sbuf)
    epst = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=epst, value=eps)
    nhet = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=nhet, value=-0.5 * eps)

    _finish_group(xa, wb, junk, epst, nhet, ov, pid, 0 * jg, jg, K, nch, cu, inv_k)
    _finish_group(xb, wb, junk, epst, nhet, ov, pid, 1 * jg, jg, K, nch, cu, inv_k)
    _finish_group(xc, wb, junk, epst, nhet, ov, pid, 2 * jg, jg, K, nch, cu, inv_k)
    _finish_group(xd, wb, junk, epst, nhet, ov, pid, 3 * jg, jg, K, nch, cu, inv_k)
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
        "groups_per_program": _NGROUPS,
        "rows_per_partition_per_group": rpp // _NGROUPS,
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
    if R % (_N_PROG * _P * _NGROUPS) != 0:
        raise ValueError("rows (batch*M) must be a multiple of %d" % (_N_PROG * _P * _NGROUPS))
    cfg = _config_for(K, R)
    out = _rmsnorm_kernel[_N_PROG](x, rms_w, float(eps), float(1.0 / K),
                                   cfg["chunk_cols"], cfg["n_chunks"], cfg["psum_chunk"])
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
