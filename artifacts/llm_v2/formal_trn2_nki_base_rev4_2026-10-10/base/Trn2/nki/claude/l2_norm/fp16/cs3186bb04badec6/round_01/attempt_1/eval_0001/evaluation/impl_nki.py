import torch

try:
    import torch_neuronx  # noqa: F401  (registers the "neuron" device if not already loaded)
except Exception:  # pragma: no cover
    pass

import nki
import nki.language as nl
import nki.isa as nisa

P = 128            # partitions
UNIT = 512         # column granularity of pieces (elements)
MAX_UNITS = 4      # piece width <= 4 * 512 = 2048 elements

_LAST_CONFIG = {}


@nki.jit
def _l2norm_main(x, eps, ng_target, npc, unit, wmax):
    # x: (N, K) in HBM. Rows are split between the programs; inside a program,
    # partition p holds R consecutive rows (contiguous R*K elements in HBM).
    N, K = x.shape
    out = nl.ndarray((N, K), dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R = N // (P * n_prog)
    L = R * K
    U = K // unit
    ng = ng_target
    if R % ng != 0:
        ng = 1
    RG = R // ng
    p0 = pid * P

    xv = x.reshape((n_prog * P, L))
    ov = out.reshape((n_prog * P, L))

    # whole per-program block stays resident in SBUF
    xt = nl.ndarray((P, L), dtype=x.dtype, buffer=nl.sbuf)
    junk = nl.ndarray((P, wmax), dtype=nl.float32, buffer=nl.sbuf)
    eps_t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=eps_t, value=eps)

    # ---- issue all loads first (row-piece granularity) ----
    for r in range(R):
        for c in range(npc):
            a = r * K + ((c * U) // npc) * unit
            b = r * K + (((c + 1) * U) // npc) * unit
            nisa.dma_copy(dst=xt[:, a:b], src=xv[p0:p0 + P, a:b])

    # ---- per group: statistics, rstd, scale, store ----
    for gi in range(ng):
        ss = nl.ndarray((P, RG), dtype=nl.float32, buffer=nl.sbuf)
        ssp = nl.ndarray((P, RG * npc), dtype=nl.float32, buffer=nl.sbuf)
        for rr in range(RG):
            r = gi * RG + rr
            for c in range(npc):
                a = r * K + ((c * U) // npc) * unit
                b = r * K + (((c + 1) * U) // npc) * unit
                if npc > 1:
                    j = rr * npc + c
                    nisa.activation(dst=junk[:, 0:b - a], op=nl.square, data=xt[:, a:b],
                                    reduce_op=nl.add, reduce_res=ssp[:, j:j + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
                else:
                    nisa.activation(dst=junk[:, 0:b - a], op=nl.square, data=xt[:, a:b],
                                    reduce_op=nl.add, reduce_res=ss[:, rr:rr + 1],
                                    reduce_cmd=nisa.reduce_cmd.reset_reduce)
            if npc > 1:
                nisa.tensor_reduce(dst=ss[:, rr:rr + 1], op=nl.add,
                                   data=ssp[:, rr * npc:(rr + 1) * npc], axis=1)

        rstd = nl.ndarray((P, RG), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rstd, op=nl.rsqrt, data=ss, bias=eps_t)

        for rr in range(RG):
            r = gi * RG + rr
            for c in range(npc):
                a = r * K + ((c * U) // npc) * unit
                b = r * K + (((c + 1) * U) // npc) * unit
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
    main_ok = (N % (P * lnc) == 0) and (K % 2 == 0) and K > 0
    R = 0
    unit = K
    npc = 1
    wmax = K
    if main_ok:
        R = N // (P * lnc)
        if K % UNIT == 0:
            unit = UNIT
            U = K // UNIT
            npc = (U + MAX_UNITS - 1) // MAX_UNITS
            wmax = ((U + npc - 1) // npc) * UNIT
        else:
            unit = K
            npc = 1
            wmax = K
        sbuf_bytes = R * K * elem_bytes + wmax * 4 + 4096
        if sbuf_bytes > 200000:
            main_ok = False
    if main_ok:
        if K <= 1024:
            ng = 1
        elif K <= 3072:
            ng = 2
        else:
            ng = 4
        while ng > 1 and R % ng != 0:
            ng //= 2
        cfg.update({
            "path": "resident_lnc2",
            "rows_per_partition": int(R),
            "groups": int(ng),
            "pieces_per_row": int(npc),
            "unit": int(unit),
            "piece_max": int(wmax),
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
    if cfg["path"] == "resident_lnc2":
        y = _l2norm_main[2](x2, eps_f, cfg["groups"], cfg["pieces_per_row"],
                            cfg["unit"], cfg["piece_max"])
    else:
        y = _l2norm_fallback(x2, eps_f)
    _LAST_CONFIG = dict(cfg)
    return y.reshape(orig_shape)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
