import torch

try:
    import torch_neuronx  # noqa: F401  (registers the "neuron" device if not already loaded)
except Exception:  # pragma: no cover
    pass

import nki
import nki.language as nl
import nki.isa as nisa

P = 128              # partitions
UNIT = 512           # column granularity (elements)
DVE_MAX_UNITS = 8    # VectorE scale piece <= 4096 elements per partition
ACT_MAX_UNITS = 10   # ScalarE square-reduce piece <= 5120 elements per partition

_LAST_CONFIG = {}


@nki.jit
def _l2norm_main(x, eps, gl, gs, ntail, nsc, nact, wact, look):
    # x: (N, K) in HBM. Rows split between the two programs; inside a program,
    # partition p holds R consecutive rows (contiguous R*K elements in HBM).
    N, K = x.shape
    out = nl.ndarray((N, K), dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R = N // (P * n_prog)
    L = R * K
    U = K // UNIT
    p0 = pid * P

    xv = x.reshape((n_prog * P, L))
    ov = out.reshape((n_prog * P, L))

    # whole per-program block resident in SBUF
    xt = nl.ndarray((P, L), dtype=x.dtype, buffer=nl.sbuf)
    junk = nl.ndarray((P, wact), dtype=nl.float32, buffer=nl.sbuf)
    ss_all = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    v_all = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    rs_all = nl.ndarray((P, R), dtype=nl.float32, buffer=nl.sbuf)
    ssp_all = nl.ndarray((P, R * nact), dtype=nl.float32, buffer=nl.sbuf)
    mh = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mh, value=-0.5)

    # ---- issue all loads first: gl whole rows per DMA (large contiguous runs) ----
    nlg = (R + gl - 1) // gl
    for j in range(nlg):
        r0 = j * gl
        r1 = min(R, r0 + gl)
        nisa.dma_copy(dst=xt[:, r0 * K:r1 * K], src=xv[p0:p0 + P, r0 * K:r1 * K])

    # ---- per-row pipeline ----
    # ScalarE computes the fp32 sum of squares `look` rows ahead (program order) of the
    # finalize (VectorE add-eps, GpSimd pow) / scale (VectorE) / grouped store (DMA).
    for r in range(R):
        h_lo = r + look
        if r == 0:
            h_lo = 0
        h_hi = min(r + look + 1, R)
        for h in range(h_lo, h_hi):
            if nact == 1:
                nisa.activation(dst=junk[:, 0:K], op=nl.square,
                                data=xt[:, h * K:(h + 1) * K],
                                reduce_op=nl.add, reduce_res=ss_all[:, h:h + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            else:
                for c in range(nact):
                    a = h * K + ((c * U) // nact) * UNIT
                    b = h * K + (((c + 1) * U) // nact) * UNIT
                    q = h * nact + c
                    nisa.activation(dst=junk[:, 0:b - a], op=nl.square, data=xt[:, a:b],
                                    reduce_op=nl.add, reduce_res=ssp_all[:, q:q + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)

        # finalize row r: sumsq (fp32) -> v = sumsq + eps -> rstd = v ** -0.5 (fp32)
        if nact > 1:
            nisa.tensor_reduce(dst=ss_all[:, r:r + 1], op=nl.add,
                               data=ssp_all[:, r * nact:(r + 1) * nact], axis=1)
        nisa.tensor_scalar(dst=v_all[:, r:r + 1], data=ss_all[:, r:r + 1], op0=nl.add,
                           operand0=eps, engine=nisa.engine.vector)
        nisa.tensor_tensor(dst=rs_all[:, r:r + 1], data1=v_all[:, r:r + 1], data2=mh,
                           op=nl.power, engine=nisa.engine.gpsimd)

        # scale (fp32 math, single cast to x.dtype), in-place
        for c in range(nsc):
            a = r * K + ((c * U) // nsc) * UNIT
            b = r * K + (((c + 1) * U) // nsc) * UNIT
            nisa.tensor_scalar(dst=xt[:, a:b], data=xt[:, a:b], op0=nl.multiply,
                               operand0=rs_all[:, r:r + 1], engine=nisa.engine.vector)

        # grouped store: gs rows per DMA, the last `ntail` rows stored individually
        s0 = r
        do_store = 0
        if r >= R - ntail:
            do_store = 1
        else:
            gi = r // gs
            s0 = gi * gs
            if r - s0 == gs - 1:
                do_store = 1
            if r == R - ntail - 1:
                do_store = 1
        if do_store == 1:
            nisa.dma_copy(dst=ov[p0:p0 + P, s0 * K:(r + 1) * K], src=xt[:, s0 * K:(r + 1) * K])
    return out


@nki.jit
def _l2norm_fallback(x, eps):
    # Generic per-128-row-tile path (only used for shapes outside the main layout).
    N, K = x.shape
    out = nl.ndarray((N, K), dtype=x.dtype, buffer=nl.shared_hbm)
    eps_t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=eps)
    junk = nl.ndarray((P, K), dtype=nl.float32, buffer=nl.sbuf)
    for i in range((N + P - 1) // P):
        r0 = i * P
        rows = min(P, N - r0)
        t = nl.ndarray((P, K), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t[0:rows, :], src=x[r0:r0 + rows, :])
        ss = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=junk[0:rows, :], op=nl.square, data=t[0:rows, :],
                        reduce_op=nl.add, reduce_res=ss[0:rows, :],
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        rs = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rs[0:rows, :], op=nl.rsqrt, data=ss[0:rows, :],
                        bias=eps_t[0:rows, :])
        nisa.tensor_scalar(dst=t[0:rows, :], data=t[0:rows, :], op0=nl.multiply,
                           operand0=rs[0:rows, :], engine=nisa.engine.vector)
        nisa.dma_copy(dst=out[r0:r0 + rows, :], src=t[0:rows, :])
    return out


def _make_config(N, K, elem_bytes):
    lnc = 2
    cfg = {"lnc": lnc, "rows": int(N), "K": int(K)}
    main_ok = (N % (P * lnc) == 0) and K > 0 and (K % UNIT == 0) and N > 0
    R = 0
    gl = 1
    gs = 2
    ntail = 2
    look = 3
    nsc = 1
    nact = 1
    wact = K
    if main_ok:
        R = N // (P * lnc)
        U = K // UNIT
        gl = 2 if K <= 2048 else 1
        gs = 2
        ntail = min(2, R)
        look = 3
        nsc = (U + DVE_MAX_UNITS - 1) // DVE_MAX_UNITS
        nact = (U + ACT_MAX_UNITS - 1) // ACT_MAX_UNITS
        wact = ((U + nact - 1) // nact) * UNIT
        sbuf_bytes = R * K * elem_bytes + wact * 4 + R * 4 * (3 + nact) + 8192
        if sbuf_bytes > 200000:
            main_ok = False
    if main_ok:
        cfg.update({
            "path": "resident_lnc2_rowpipe_bigdma",
            "rows_per_partition": int(R),
            "load_rows_per_dma": int(gl),
            "store_rows_per_dma": int(gs),
            "store_tail_single_rows": int(ntail),
            "act_lookahead_rows": int(look),
            "dve_pieces_per_row": int(nsc),
            "act_pieces_per_row": int(nact),
            "act_piece_max": int(wact),
            "rstd": "vector_add_eps+gpsimd_pow",
        })
    else:
        cfg.update({"path": "tiled_fallback", "tile_rows": P})
    return cfg


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    global _LAST_CONFIG
    orig_shape = x.shape
    K = int(orig_shape[-1])
    N = int(x.numel() // K)
    if not x.is_contiguous():
        x = x.contiguous()
    x2 = x.reshape(N, K)
    cfg = _make_config(N, K, x.element_size())
    eps_f = float(eps)
    if cfg["path"] == "resident_lnc2_rowpipe_bigdma":
        y = _l2norm_main[2](x2, eps_f, cfg["load_rows_per_dma"], cfg["store_rows_per_dma"],
                            cfg["store_tail_single_rows"], cfg["dve_pieces_per_row"],
                            cfg["act_pieces_per_row"], cfg["act_piece_max"],
                            cfg["act_lookahead_rows"])
    else:
        y = _l2norm_fallback(x2, eps_f)
    _LAST_CONFIG = dict(cfg)
    return y.reshape(orig_shape)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
