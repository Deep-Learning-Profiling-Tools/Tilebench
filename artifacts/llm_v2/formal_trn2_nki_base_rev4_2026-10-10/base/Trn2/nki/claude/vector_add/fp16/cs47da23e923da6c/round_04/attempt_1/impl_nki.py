import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
_ONE_MI = 1 << 20
_POOL_BYTES_PER_PARTITION = 131072   # SBUF budget per partition for the x / y / out slot pools

_last_config = {}


@nki.jit
def vector_add_kernel(x, y, F, NB, CW):
    # x, y: (n,) fp16 in HBM; n is a multiple of num_programs * 128 * F (guaranteed by run()).
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    T_total = n // (PMAX * F)          # total (128, F) tiles
    T = T_total // n_prog              # tiles per program (physical core)
    t0 = pid * T                       # contiguous block of tiles for this program

    x3 = x.reshape((T_total, PMAX, F))
    y3 = y.reshape((T_total, PMAX, F))
    o3 = out.reshape((T_total, PMAX, F))

    # NB slots each for x, y and the result: loads only wait on the add that consumed the
    # slot (never on a store), stores only wait on the add that produced them.
    xb = nl.ndarray((PMAX, NB * F), dtype=x.dtype, buffer=nl.sbuf)
    yb = nl.ndarray((PMAX, NB * F), dtype=x.dtype, buffer=nl.sbuf)
    ob = nl.ndarray((PMAX, NB * F), dtype=x.dtype, buffer=nl.sbuf)

    D = min(NB, T)
    NCH = F // CW                      # chunks used for the final tile (short tail)

    # prologue: prefetch tiles 0..D-1 (tile t -> slot t)
    for t in range(D):
        lo = t * F
        g = t0 + t
        if t == T - 1:
            for c in range(NCH):
                a = lo + c * CW
                nisa.dma_copy(dst=xb[:, a:a + CW], src=x3[g, :, c * CW:(c + 1) * CW])
                nisa.dma_copy(dst=yb[:, a:a + CW], src=y3[g, :, c * CW:(c + 1) * CW])
        else:
            nisa.dma_copy(dst=xb[:, lo:lo + F], src=x3[g, :, :])
            nisa.dma_copy(dst=yb[:, lo:lo + F], src=y3[g, :, :])

    s = 0
    for t in range(T):
        lo = s * F
        g = t0 + t

        if t == T - 1:
            for c in range(NCH):
                a = lo + c * CW
                nisa.tensor_tensor(dst=ob[:, a:a + CW], data1=xb[:, a:a + CW],
                                   data2=yb[:, a:a + CW], op=nl.add,
                                   engine=nisa.engine.vector)
                nisa.dma_copy(dst=o3[g, :, c * CW:(c + 1) * CW], src=ob[:, a:a + CW])
        else:
            nisa.tensor_tensor(dst=ob[:, lo:lo + F], data1=xb[:, lo:lo + F],
                               data2=yb[:, lo:lo + F], op=nl.add,
                               engine=nisa.engine.vector)
            nisa.dma_copy(dst=o3[g, :, :], src=ob[:, lo:lo + F])

        # refill this x/y slot with tile t + D (slot matches since D == NB whenever tn < T)
        tn = t + D
        if tn < T:
            gn = t0 + tn
            if tn == T - 1:
                for c in range(NCH):
                    a = lo + c * CW
                    nisa.dma_copy(dst=xb[:, a:a + CW], src=x3[gn, :, c * CW:(c + 1) * CW])
                    nisa.dma_copy(dst=yb[:, a:a + CW], src=y3[gn, :, c * CW:(c + 1) * CW])
            else:
                nisa.dma_copy(dst=xb[:, lo:lo + F], src=x3[gn, :, :])
                nisa.dma_copy(dst=yb[:, lo:lo + F], src=y3[gn, :, :])

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
    if k >= 4:
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
    cap = _POOL_BYTES_PER_PARTITION // (6 * F)     # 3 pools x 2 bytes x F per slot
    cap = max(1, min(8, cap))
    NB = max(1, min(T, cap))
    CW = min(512, F)
    return {
        "path": "pipelined_3pool",
        "programs": n_prog,
        "F": F,
        "tile_elems": PMAX * F,
        "tiles_per_program": T,
        "NB": NB,
        "tail_chunk_cols": CW,
    }


def run(x, y):
    global _last_config
    n = x.shape[0]
    cfg = _choose_config(n)
    if cfg["path"] == "pipelined_3pool":
        out = vector_add_kernel(x, y, cfg["F"], cfg["NB"], cfg["tail_chunk_cols"])
    else:
        out = vector_add_generic(x, y, cfg["F"])
    _last_config = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_last_config)
