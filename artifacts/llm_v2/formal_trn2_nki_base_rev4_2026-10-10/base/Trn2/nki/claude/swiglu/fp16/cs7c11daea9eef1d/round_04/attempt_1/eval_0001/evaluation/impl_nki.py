import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_DEPTH = 3          # load prefetch distance (items) in the explicit pipeline
_LAST_CONFIG = {}


def _load(vb, vs, k, nbig, t0, sbase, f, fs, dt):
    # Load item k of this program: big tile (128 x f) or tail sub-tile (128 x fs).
    P = 128
    w = f
    if k >= nbig:
        w = fs
    t = nl.ndarray((P, w), dtype=dt, buffer=nl.sbuf)
    if k < nbig:
        nisa.dma_copy(dst=t, src=vb[t0 + k, :, 0:f])
    else:
        nisa.dma_copy(dst=t, src=vs[sbase + k - nbig, :, 0:fs])
    return t


def _silu(xt, k, nbig, f, fs):
    # silu(x) in fp32 on ScalarE (input upcast to fp32, fp32 result tile)
    P = 128
    w = f
    if k >= nbig:
        w = fs
    st = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=st, op=nl.silu, data=xt)
    return st


def _mul(st, yt, k, nbig, f, fs, dt):
    # fp32 multiply by y on VectorE, single downcast to the output dtype
    P = 128
    w = f
    if k >= nbig:
        w = fs
    ot = nl.ndarray((P, w), dtype=dt, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=ot, data1=st, data2=yt, op=nl.multiply,
                       engine=nisa.engine.vector)
    return ot


def _store(ot, ob, osm, k, nbig, t0, sbase, f, fs):
    if k < nbig:
        nisa.dma_copy(dst=ob[t0 + k, :, 0:f], src=ot)
    else:
        nisa.dma_copy(dst=osm[sbase + k - nbig, :, 0:fs], src=ot)


@nki.jit
def _swiglu_kernel(x, y, f, r, tail_split):
    M, N = x.shape
    P = 128
    dt = x.dtype
    out = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)

    E = M * N
    TB = E // (P * f)          # number of big contiguous tiles (128 x f each)
    fs = f // r                # tail sub-tile width

    xb = x.reshape((TB, P, f))
    yb = y.reshape((TB, P, f))
    ob = out.reshape((TB, P, f))
    xs = x.reshape((TB * r, P, fs))
    ys = y.reshape((TB * r, P, fs))
    osm = out.reshape((TB * r, P, fs))

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    per = TB
    t0 = 0
    if TB % n_prog == 0:
        per = TB // n_prog
        t0 = pid * per

    nbig = per
    nsmall = 0
    if tail_split > 0:
        if per >= 2:
            nbig = per - 1
            nsmall = r
    n = nbig + nsmall
    sbase = (t0 + nbig) * r

    # ---- prologue: prefetch items 0..2, silu of item 0 ----
    xc = _load(xb, xs, 0, nbig, t0, sbase, f, fs, dt)
    yc = _load(yb, ys, 0, nbig, t0, sbase, f, fs, dt)
    x1 = xc
    y1 = yc
    x2 = xc
    y2 = yc
    x3 = xc
    y3 = yc
    if n > 1:
        x1 = _load(xb, xs, 1, nbig, t0, sbase, f, fs, dt)
        y1 = _load(yb, ys, 1, nbig, t0, sbase, f, fs, dt)
    if n > 2:
        x2 = _load(xb, xs, 2, nbig, t0, sbase, f, fs, dt)
        y2 = _load(yb, ys, 2, nbig, t0, sbase, f, fs, dt)
    sc = _silu(xc, 0, nbig, f, fs)
    sn = sc
    op = sc

    # ---- steady state: load k+3, silu k+1, multiply k, store k-1 ----
    for k in range(n):
        if k + 3 < n:
            x3 = _load(xb, xs, k + 3, nbig, t0, sbase, f, fs, dt)
            y3 = _load(yb, ys, k + 3, nbig, t0, sbase, f, fs, dt)
        if k + 1 < n:
            sn = _silu(x1, k + 1, nbig, f, fs)
        oc = _mul(sc, yc, k, nbig, f, fs, dt)
        if k > 0:
            _store(op, ob, osm, k - 1, nbig, t0, sbase, f, fs)
        op = oc
        yc = y1
        y1 = y2
        y2 = y3
        x1 = x2
        x2 = x3
        sc = sn

    # ---- epilogue: store the last item ----
    _store(op, ob, osm, n - 1, nbig, t0, sbase, f, fs)
    return out


def run(x, y):
    global _LAST_CONFIG
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.dim() == 2, "expected 2-D inputs (M, N)"
    M, N = int(x.shape[0]), int(x.shape[1])
    E = M * N
    assert E % _P == 0, "M*N must be a multiple of 128"

    n_prog = 2  # bare kernel call runs with LNC=2 on this host
    W = E // _P
    wp = W // n_prog if W % n_prog == 0 else W

    # deterministic tile choice: 1 MiB fp16 DMA tiles when there are >= 16 per program
    f = 4096 if wp >= 65536 else 2048
    while f > 1 and E % (n_prog * _P * f) != 0:
        f //= 2
    if E % (_P * f) != 0:
        f = 1

    r = 1
    while r < 4 and (f // (r * 2)) >= 256:
        r *= 2

    TB = E // (_P * f)
    per = TB // n_prog if TB % n_prog == 0 else TB
    tail_split = 1 if (r > 1 and per >= 2) else 0
    n_items = (per - 1 + r) if tail_split else per

    out = _swiglu_kernel(x, y, f, r, tail_split)

    _LAST_CONFIG = {
        "layout": "contiguous tiles: view (T,128,f), program-halves",
        "pipeline": "explicit: load k+3 | silu k+1 | mul k | store k-1",
        "prefetch_depth": _DEPTH,
        "partitions": _P,
        "programs": n_prog,
        "elems_per_partition_per_program": wp,
        "tile_free": f,
        "tiles_per_program": per,
        "items_per_program": n_items,
        "tail_split": tail_split,
        "tail_subtiles": r if tail_split else 1,
        "tail_subtile_free": (f // r) if tail_split else f,
        "silu_engine": "scalar(activation silu, fp32 out, double-buffered)",
        "mul_engine": "vector(tensor_tensor, fp32 math, fp16 out tile)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
