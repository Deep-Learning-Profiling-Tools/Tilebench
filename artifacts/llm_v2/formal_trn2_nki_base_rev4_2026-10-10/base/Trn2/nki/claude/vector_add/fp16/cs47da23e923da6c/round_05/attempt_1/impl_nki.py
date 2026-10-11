import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
_POOL_BYTES_PER_PARTITION = 163840   # SBUF budget (per partition) for the x+y slot pools
_ONE_MI = 1 << 20

_last_config = {}


@nki.jit
def vector_add_kernel(x, y, F, NB, LC):
    # x, y: (n,) in HBM. n must be a multiple of num_programs * 128 * F (guaranteed by run()).
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    T_total = n // (PMAX * F)          # total (128, F) tiles
    T = T_total // n_prog              # tiles per program (physical core)
    t0 = pid * T                       # first tile of this program (contiguous block)

    x3 = x.reshape((T_total, PMAX, F))
    y3 = y.reshape((T_total, PMAX, F))
    o3 = out.reshape((T_total, PMAX, F))

    # explicit multi-buffer pool: NB slots of (128, F) for x (also holds the sum) and y
    xb = nl.ndarray((PMAX, NB * F), dtype=x.dtype, buffer=nl.sbuf)
    yb = nl.ndarray((PMAX, NB * F), dtype=x.dtype, buffer=nl.sbuf)

    D = min(NB, T)

    # prologue: prefetch the first D tiles (tile t -> slot t)
    for t in range(D):
        lo = t * F
        nisa.dma_copy(dst=xb[:, lo:lo + F], src=x3[t0 + t, :, :])
        nisa.dma_copy(dst=yb[:, lo:lo + F], src=y3[t0 + t, :, :])

    s = 0
    ps = NB - 1
    for t in range(T):
        lo = s * F
        hi = lo + F

        nch = 1
        if t == T - 1:
            nch = LC                    # split the last tile to shorten the exposed tail
        CW = F // nch
        for c in range(nch):
            a = lo + c * CW
            b = a + CW
            nisa.tensor_tensor(dst=xb[:, a:b], data1=xb[:, a:b], data2=yb[:, a:b],
                               op=nl.add, engine=nisa.engine.vector)
            nisa.dma_copy(dst=o3[t0 + t, :, c * CW:(c + 1) * CW], src=xb[:, a:b])

        # refill: y slot of this tile right away (only waits on the add);
        # x slot of the previous tile one iteration later (waits on that tile's store).
        # Refills only happen when D == NB, so slot indices match tile % NB.
        tn = t + D
        if tn < T:
            nisa.dma_copy(dst=yb[:, lo:hi], src=y3[t0 + tn, :, :])
        if D >= 2:
            if t >= 1:
                tx = t - 1 + D
                if tx < T:
                    plo = ps * F
                    nisa.dma_copy(dst=xb[:, plo:plo + F], src=x3[t0 + tx, :, :])
        else:
            if tn < T:
                nisa.dma_copy(dst=xb[:, lo:hi], src=x3[t0 + tn, :, :])

        ps = s
        s = s + 1
        if s == NB:
            s = 0

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


def _choose_config(n):
    n_prog = 2
    k = n // _ONE_MI
    if k >= 8:
        F = 4096
    elif k >= 2:
        F = 2048
    else:
        F = 1024
    while F > 256 and n % (n_prog * PMAX * F) != 0:
        F //= 2
    if n <= 0 or n % (n_prog * PMAX * F) != 0:
        return {"path": "generic", "F": 2048}
    T = n // (n_prog * PMAX * F)
    cap = max(2, _POOL_BYTES_PER_PARTITION // (4 * F))   # 2 pools x 2 bytes x F per slot
    NB = max(1, min(T, cap))
    LC = F // 1024 if F >= 2048 else 1
    return {
        "path": "pipelined",
        "programs": n_prog,
        "F": F,
        "tile_elems": PMAX * F,
        "tiles_per_program": T,
        "NB": NB,
        "last_tile_chunks": LC,
        "x_refill_lag": 1,
    }


def run(x, y):
    global _last_config
    n = x.shape[0]
    cfg = _choose_config(n)
    if cfg["path"] == "pipelined":
        out = vector_add_kernel(x, y, cfg["F"], cfg["NB"], cfg["last_tile_chunks"])
    else:
        out = vector_add_generic(x, y, cfg["F"])
    _last_config = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_last_config)
