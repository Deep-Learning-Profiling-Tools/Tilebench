import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
FMAX = 4096            # max columns per segment: (128, 4096) fp16 = 1 MiB per DMA
NB_SLOTS = 6           # x / y input ring slots (each FMAX columns)
NO_SLOTS = 6           # output ring slots (each FMAX columns)
_UNIT = 2 * PMAX * FMAX   # n must be a multiple of this for the pipelined path (2 halves x 128 x 4096)

_last_config = {}


@nki.jit
def vector_add_kernel(x, y, SW, SO, NB, NO):
    # x, y: (n,) in HBM. The data is split into 2 contiguous halves (one per program / physical core).
    # Each half is a sequence of contiguous segments; segment i holds 128*SW[i] elements starting at
    # column offset SO[i] (element offset 128*SO[i] inside the half), viewed as a (128, SW[i]) tile.
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    halves = 2 // n_prog
    half = n // 2
    nseg = len(SW)

    xb = nl.ndarray((PMAX, NB * FMAX), dtype=x.dtype, buffer=nl.sbuf)
    yb = nl.ndarray((PMAX, NB * FMAX), dtype=x.dtype, buffer=nl.sbuf)
    ob = nl.ndarray((PMAX, NO * FMAX), dtype=x.dtype, buffer=nl.sbuf)

    D = min(NB, nseg)

    for h in range(halves):
        base = (pid * halves + h) * half

        # prologue: prefetch the first D segments (segment j -> input slot j)
        for j in range(D):
            wj = SW[j]
            loj = j * FMAX
            ntj = n // (PMAX * wj)
            tj = (base + PMAX * SO[j]) // (PMAX * wj)
            xj = x.reshape((ntj, PMAX, wj))
            yj = y.reshape((ntj, PMAX, wj))
            nisa.dma_copy(dst=xb[:, loj:loj + wj], src=xj[tj, :, :])
            nisa.dma_copy(dst=yb[:, loj:loj + wj], src=yj[tj, :, :])

        for i in range(nseg):
            w = SW[i]
            lo = (i - (i // NB) * NB) * FMAX
            so = (i - (i // NO) * NO) * FMAX

            nisa.tensor_tensor(dst=ob[:, so:so + w], data1=xb[:, lo:lo + w],
                               data2=yb[:, lo:lo + w], op=nl.add, engine=nisa.engine.vector)

            nt = n // (PMAX * w)
            t = (base + PMAX * SO[i]) // (PMAX * w)
            o3 = out.reshape((nt, PMAX, w))
            nisa.dma_copy(dst=o3[t, :, :], src=ob[:, so:so + w])

            # refill this input slot with segment i + D (only when D == NB, so the slot matches)
            k = i + D
            if k < nseg:
                wk = SW[k]
                ntk = n // (PMAX * wk)
                tk = (base + PMAX * SO[k]) // (PMAX * wk)
                xk = x.reshape((ntk, PMAX, wk))
                yk = y.reshape((ntk, PMAX, wk))
                nisa.dma_copy(dst=xb[:, lo:lo + wk], src=xk[tk, :, :])
                nisa.dma_copy(dst=yb[:, lo:lo + wk], src=yk[tk, :, :])

    return out


@nki.jit
def vector_add_generic(x, y, F):
    # Fallback for lengths outside the configured domain (arbitrary n); partial tiles via slices.
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    q = n // F
    r = n - q * F
    rr = 0
    while rr < q:
        rows = min(PMAX, q - rr)
        off = rr * F
        a = nl.ndarray((PMAX, F), dtype=x.dtype, buffer=nl.sbuf)
        b = nl.ndarray((PMAX, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a[0:rows, :], src=x.ap(pattern=[[F, rows], [1, F]], offset=off))
        nisa.dma_copy(dst=b[0:rows, :], src=y.ap(pattern=[[F, rows], [1, F]], offset=off))
        nisa.tensor_tensor(dst=a[0:rows, :], data1=a[0:rows, :], data2=b[0:rows, :], op=nl.add)
        nisa.dma_copy(dst=out.ap(pattern=[[F, rows], [1, F]], offset=off), src=a[0:rows, :])
        rr = rr + rows
    if r > 0:
        off = q * F
        a2 = nl.ndarray((1, F), dtype=x.dtype, buffer=nl.sbuf)
        b2 = nl.ndarray((1, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a2[0:1, 0:r], src=x.ap(pattern=[[r, 1], [1, r]], offset=off))
        nisa.dma_copy(dst=b2[0:1, 0:r], src=y.ap(pattern=[[r, 1], [1, r]], offset=off))
        nisa.tensor_tensor(dst=a2[0:1, 0:r], data1=a2[0:1, 0:r], data2=b2[0:1, 0:r], op=nl.add)
        nisa.dma_copy(dst=out.ap(pattern=[[r, 1], [1, r]], offset=off), src=a2[0:1, 0:r])
    return out


def _segments(C):
    # C = columns per half (multiple of FMAX). Ramp-up head, FMAX body, ramp-down tail.
    kk = C // FMAX
    if kk <= 1:
        ws = [1024, 1024, 1024, 512, 256, 256]
    else:
        ws = [1024, 1024, 2048]
        for _ in range(kk - 2):
            ws.append(FMAX)
        ws.extend([2048, 1024, 512, 256, 256])
    offs = []
    acc = 0
    for w in ws:
        offs.append(acc)
        acc += w
    return ws, offs, acc


def _choose_config(n):
    if n <= 0 or n % _UNIT != 0:
        return {"path": "generic", "F": 2048}
    C = n // (2 * PMAX)
    ws, offs, total = _segments(C)
    ok = (total == C)
    for w, o in zip(ws, offs):
        if w > FMAX or o % w != 0 or FMAX % w != 0:
            ok = False
    if not ok:
        return {"path": "generic", "F": 2048}
    return {
        "path": "pipelined",
        "programs": 2,
        "fmax": FMAX,
        "NB": NB_SLOTS,
        "NO": NO_SLOTS,
        "num_segments": len(ws),
        "segment_widths": list(ws),
        "segment_col_offsets": list(offs),
    }


def run(x, y):
    global _last_config
    n = x.shape[0]
    cfg = _choose_config(n)
    if cfg["path"] == "pipelined":
        out = vector_add_kernel(x, y, tuple(cfg["segment_widths"]), tuple(cfg["segment_col_offsets"]),
                                cfg["NB"], cfg["NO"])
    else:
        out = vector_add_generic(x, y, cfg["F"])
    _last_config = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_last_config)
