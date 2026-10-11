import torch
import nki
import nki.language as nl
import nki.isa as nisa

# LayerNorm over the last dim, fp16 in/out, fp32 statistics and apply.
# Layout: the (N, K) rows are viewed as (2*128, rpp*K): every partition owns
# rpp consecutive rows; LNC=2 programs each own 128 partitions.
# Per group (R rows per partition) the pipeline is:
#   DMA load -> [VectorE row sum | ScalarE square+reduce] -> tiny fp32 stats
#   -> ScalarE (x*rstd - mean*rstd) fp32 -> VectorE *W fp32 -> GpSimd +B -> fp16
#   -> DMA store, software-pipelined over 3 stages (load / stats / apply).

_P = 128
_BLK = 512
_CHUNK_BLKS = 4          # max columns per compute instruction = 2048
_SMALL_K = 1024          # K <= this: 2 rows per partition per group
_NBUF = 3

_LAST_CONFIG = {}


def _pick3(i, a, b, c):
    r = c
    if i == 0:
        r = a
    elif i == 1:
        r = b
    return r


def _chunk_lo(c, nc, nblk, k):
    v = ((c * nblk) // nc) * 512
    if v > k:
        v = k
    return v


@nki.jit
def _layernorm_kernel(x, w, b, n_rows, k_cols, eps, rows_per_group, n_chunks, chunk_cols):
    P = 128
    K = k_cols
    R = rows_per_group
    NC = n_chunks
    CWB = chunk_cols

    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    rpp = n_rows // (n_prog * P)
    G = rpp // R
    F = R * K
    nblk = (K + 511) // 512
    inv_k = 1.0 / K
    neg_inv_k = -1.0 / K
    p0 = pid * P

    xr = x.reshape((n_prog * P, rpp * K))
    orr = out.reshape((n_prog * P, rpp * K))

    # weight / bias broadcast to all partitions (fp16 values, exact)
    w2 = w.reshape((1, K))
    b2 = b.reshape((1, K))
    wt = nl.ndarray((P, F), dtype=w.dtype, buffer=nl.sbuf)
    bt = nl.ndarray((P, F), dtype=b.dtype, buffer=nl.sbuf)
    for j in range(R):
        nisa.dma_copy(dst=wt[:, j * K:(j + 1) * K], src=w2.ap(pattern=[[0, P], [1, K]], offset=0))
        nisa.dma_copy(dst=bt[:, j * K:(j + 1) * K], src=b2.ap(pattern=[[0, P], [1, K]], offset=0))

    mhalf = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mhalf, value=-0.5)
    junk = nl.ndarray((P, CWB), dtype=nl.bfloat16, buffer=nl.sbuf)

    xb0 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb1 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    xb2 = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    rb0 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    rb1 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    rb2 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nb0 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nb1 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    nb2 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    fb0 = nl.ndarray((P, CWB), dtype=nl.float32, buffer=nl.sbuf)
    fb1 = nl.ndarray((P, CWB), dtype=nl.float32, buffer=nl.sbuf)
    fb2 = nl.ndarray((P, CWB), dtype=nl.float32, buffer=nl.sbuf)
    yb0 = nl.ndarray((P, CWB), dtype=x.dtype, buffer=nl.sbuf)
    yb1 = nl.ndarray((P, CWB), dtype=x.dtype, buffer=nl.sbuf)
    yb2 = nl.ndarray((P, CWB), dtype=x.dtype, buffer=nl.sbuf)

    for s in range(G + 2):
        s1 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        s2 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
        s1p = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)
        s2p = nl.ndarray((P, NC), dtype=nl.float32, buffer=nl.sbuf)

        # ---- stage A: load group s ----
        if s < G:
            xa = _pick3(s % 3, xb0, xb1, xb2)
            ca = s * F
            if NC > 1:
                for c in range(NC):
                    a0 = _chunk_lo(c, NC, nblk, K)
                    a1 = _chunk_lo(c + 1, NC, nblk, K)
                    nisa.dma_copy(dst=xa[:, a0:a1], src=xr[p0:p0 + P, ca + a0:ca + a1])
            else:
                nisa.dma_copy(dst=xa[:, 0:F], src=xr[p0:p0 + P, ca:ca + F])

        # ---- stage B1: row reductions (fp32) for group s-1 ----
        if s >= 1:
            if s <= G:
                gb = s - 1
                xs = _pick3(gb % 3, xb0, xb1, xb2)
                if NC > 1:
                    for c in range(NC):
                        a0 = _chunk_lo(c, NC, nblk, K)
                        a1 = _chunk_lo(c + 1, NC, nblk, K)
                        nisa.tensor_reduce(dst=s1p[:, c:c + 1], op=nl.add, data=xs[:, a0:a1], axis=1)
                        nisa.activation(dst=junk[:, 0:a1 - a0], op=nl.square, data=xs[:, a0:a1],
                                        reduce_op=nl.add, reduce_res=s2p[:, c:c + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
                else:
                    for j in range(R):
                        nisa.tensor_reduce(dst=s1[:, j:j + 1], op=nl.add,
                                           data=xs[:, j * K:(j + 1) * K], axis=1)
                        nisa.activation(dst=junk[:, 0:K], op=nl.square, data=xs[:, j * K:(j + 1) * K],
                                        reduce_op=nl.add, reduce_res=s2[:, j:j + 1],
                                        reduce_cmd=nisa.reduce_cmd.reset_reduce)

        # ---- stage C: apply group s-2 ----
        if s >= 2:
            gc = s - 2
            xc = _pick3(gc % 3, xb0, xb1, xb2)
            rsc = _pick3(gc % 3, rb0, rb1, rb2)
            nmc = _pick3(gc % 3, nb0, nb1, nb2)
            cc0 = gc * F
            if R == 1:
                for c in range(NC):
                    a0 = _chunk_lo(c, NC, nblk, K)
                    a1 = _chunk_lo(c + 1, NC, nblk, K)
                    cw = a1 - a0
                    q = (gc * NC + c) % 3
                    ft = _pick3(q, fb0, fb1, fb2)
                    yt = _pick3(q, yb0, yb1, yb2)
                    nisa.activation(dst=ft[:, 0:cw], op=nl.copy, data=xc[:, a0:a1],
                                    scale=rsc, bias=nmc)
                    nisa.tensor_tensor(dst=ft[:, 0:cw], data1=ft[:, 0:cw], data2=wt[:, a0:a1],
                                       op=nl.multiply, engine=nisa.engine.vector)
                    nisa.tensor_tensor(dst=yt[:, 0:cw], data1=ft[:, 0:cw], data2=bt[:, a0:a1],
                                       op=nl.add, engine=nisa.engine.gpsimd)
                    nisa.dma_copy(dst=orr[p0:p0 + P, cc0 + a0:cc0 + a1], src=yt[:, 0:cw])
            else:
                q = gc % 3
                ft = _pick3(q, fb0, fb1, fb2)
                yt = _pick3(q, yb0, yb1, yb2)
                for j in range(R):
                    nisa.activation(dst=ft[:, j * K:(j + 1) * K], op=nl.copy,
                                    data=xc[:, j * K:(j + 1) * K],
                                    scale=rsc[:, j:j + 1], bias=nmc[:, j:j + 1])
                nisa.tensor_tensor(dst=ft[:, 0:F], data1=ft[:, 0:F], data2=wt[:, 0:F],
                                   op=nl.multiply, engine=nisa.engine.vector)
                nisa.tensor_tensor(dst=yt[:, 0:F], data1=ft[:, 0:F], data2=bt[:, 0:F],
                                   op=nl.add, engine=nisa.engine.gpsimd)
                nisa.dma_copy(dst=orr[p0:p0 + P, cc0:cc0 + F], src=yt[:, 0:F])

        # ---- stage B2: finalize statistics of group s-1 ----
        if s >= 1:
            if s <= G:
                gb = s - 1
                rs = _pick3(gb % 3, rb0, rb1, rb2)
                nm = _pick3(gb % 3, nb0, nb1, nb2)
                if NC > 1:
                    nisa.tensor_reduce(dst=s1, op=nl.add, data=s1p, axis=1)
                    nisa.tensor_reduce(dst=s2, op=nl.add, data=s2p, axis=1)
                negmean = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=negmean, data=s1, op0=nl.multiply, operand0=neg_inv_k)
                msq = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=msq, data1=negmean, data2=negmean, op=nl.multiply)
                vr = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=vr, data=s2, op0=nl.multiply, operand0=inv_k)
                nisa.tensor_tensor(dst=vr, data1=vr, data2=msq, op=nl.subtract)
                nisa.tensor_scalar(dst=vr, data=vr, op0=nl.maximum, operand0=0.0,
                                   op1=nl.add, operand1=eps)
                # rstd = (var + eps) ^ -0.5, refined by one Newton step
                r0 = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=r0, data1=vr, data2=mhalf, op=nl.power,
                                   engine=nisa.engine.gpsimd)
                tn = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=tn, data1=r0, data2=r0, op=nl.multiply)
                nisa.tensor_tensor(dst=tn, data1=tn, data2=vr, op=nl.multiply)
                nisa.tensor_scalar(dst=tn, data=tn, op0=nl.multiply, operand0=-0.5,
                                   op1=nl.add, operand1=1.5)
                nisa.tensor_tensor(dst=rs, data1=r0, data2=tn, op=nl.multiply)
                nisa.tensor_tensor(dst=nm, data1=negmean, data2=rs, op=nl.multiply)

    return out


def _make_config(n_rows, k):
    rpp = n_rows // (2 * _P)
    nblk = (k + _BLK - 1) // _BLK
    if k <= _SMALL_K and rpp % 2 == 0:
        R = 2
        NC = 1
        cwb = 2 * k
    else:
        R = 1
        NC = (nblk + _CHUNK_BLKS - 1) // _CHUNK_BLKS
        cwb = ((nblk + NC - 1) // NC) * _BLK
        if cwb > k:
            cwb = k
    return {
        "rows_per_group": R,
        "n_chunks": NC,
        "chunk_buf_cols": cwb,
        "max_chunk_cols": _CHUNK_BLKS * _BLK,
        "small_k_threshold": _SMALL_K,
        "x_buffers": _NBUF,
        "lnc_programs": 2,
        "partitions": _P,
    }


def run(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    global _LAST_CONFIG
    if not x.is_contiguous():
        x = x.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    K = int(x.shape[-1])
    n_rows = int(x.numel() // K)
    assert n_rows % (2 * _P) == 0, "row count must be a multiple of 256"
    cfg = _make_config(n_rows, K)
    out = _layernorm_kernel(x, weight, bias, n_rows, K, float(eps),
                            cfg["rows_per_group"], cfg["n_chunks"], cfg["chunk_buf_cols"])
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
