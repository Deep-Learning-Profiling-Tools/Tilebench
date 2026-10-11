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
DMA_MAX_UNITS = 8    # DMA piece <= 4096 elements per partition
ACT_MAX_UNITS = 10   # ScalarE square-reduce piece <= 5120 elements per partition

_LAST_CONFIG = {}


@nki.jit
def _l2norm_main(x, eps, G, npc, nact, wact):
    # x: (N, K) in HBM. Rows split between the two programs; inside a program,
    # partition p holds R consecutive rows (contiguous R*K elements in HBM).
    N, K = x.shape
    out = nl.ndarray((N, K), dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R = N // (P * n_prog)
    L = R * K
    U = K // UNIT
    ng = R // G
    p0 = pid * P

    xv = x.reshape((n_prog * P, L))
    ov = out.reshape((n_prog * P, L))

    # whole per-program block resident in SBUF
    xt = nl.ndarray((P, L), dtype=x.dtype, buffer=nl.sbuf)
    junk = nl.ndarray((P, wact), dtype=nl.float32, buffer=nl.sbuf)
    mh = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mh, value=-0.5)

    # ---- issue all loads first ----
    for r in range(R):
        for c in range(npc):
            a = r * K + ((c * U) // npc) * UNIT
            b = r * K + (((c + 1) * U) // npc) * UNIT
            nisa.dma_copy(dst=xt[:, a:b], src=xv[p0:p0 + P, a:b])

    # ---- per row group: fp32 sum of squares (ScalarE), rstd (VectorE + GpSimd),
    #      scale (VectorE, fp32 math, single cast to fp16), store ----
    for g in range(ng):
        ss = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
        for rr in range(G):
            r = g * G + rr
            if nact == 1:
                nisa.activation(dst=junk[:, 0:K], op=nl.square,
                                data=xt[:, r * K:(r + 1) * K],
                                reduce_op=nl.add, reduce_res=ss[:, rr:rr + 1],
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
            else:
                ssp = nl.ndarray((P, nact), dtype=nl.float32, buffer=nl.sbuf)
                for c in range(nact):
                    a = r * K + ((c * U) // nact) * UNIT
                    b = r * K + (((c + 1) * U) // nact) * UNIT
                    nisa.activation(dst=junk[:, 0:b - a], op=nl.square, data=xt[:, a:b],
                                    reduce_op=nl.add, reduce_res=ssp[:, c:c + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                nisa.tensor_reduce(dst=ss[:, rr:rr + 1], op=nl.add, data=ssp, axis=1)

        # v = sumsq + eps (fp32), rstd = v ** -0.5 (fp32)
        v = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=v, data=ss, op0=nl.add, operand0=eps,
                           engine=nisa.engine.vector)
        rstd = nl.ndarray((P, G), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=rstd, data1=v, data2=mh, op=nl.power,
                           engine=nisa.engine.gpsimd)

        for rr in range(G):
            r = g * G + rr
            for c in range(npc):
                a = r * K + ((c * U) // npc) * UNIT
                b = r * K + (((c + 1) * U) // npc) * UNIT
                nisa.tensor_scalar(dst=xt[:, a:b], data=xt[:, a:b], op0=nl.multiply,
                                   operand0=rstd[:, rr:rr + 1], engine=nisa.engine.vector)
                nisa.dma_copy(dst=ov[p0:p0 + P, a:b], src=xt[:, a:b])
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
    main_ok = (N % (P * lnc) == 0) and K > 0 and (K % UNIT == 0)
    R = 0
    npc = 1
    nact = 1
    wact = K
    G = 1
    if main_ok:
        R = N // (P * lnc)
        U = K // UNIT
        npc = (U + DMA_MAX_UNITS - 1) // DMA_MAX_UNITS
        nact = (U + ACT_MAX_UNITS - 1) // ACT_MAX_UNITS
        wact = ((U + nact - 1) // nact) * UNIT
        G = 2 if K <= 1024 else 1
        if R % G != 0 or nact > 1:
            G = 1
        sbuf_bytes = R * K * elem_bytes + wact * 4 + 8192
        if sbuf_bytes > 200000:
            main_ok = False
    if main_ok:
        cfg.update({
            "path": "resident_lnc2_pipelined",
            "rows_per_partition": int(R),
            "rows_per_group": int(G),
            "dma_pieces_per_row": int(npc),
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
    if cfg["path"] == "resident_lnc2_pipelined":
        y = _l2norm_main[2](x2, eps_f, cfg["rows_per_group"], cfg["dma_pieces_per_row"],
                            cfg["act_pieces_per_row"], cfg["act_piece_max"])
    else:
        y = _l2norm_fallback(x2, eps_f)
    _LAST_CONFIG = dict(cfg)
    return y.reshape(orig_shape)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
