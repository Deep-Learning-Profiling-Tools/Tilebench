import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_CW = 1024          # compute chunk width (columns) inside one DMA tile
_LAST_CONFIG = {}


def _gate_tile(xv, yv, ov, t, w, cw, dt):
    # One fused tile: out = silu(x) * y, fp32 math, single downcast.
    # xv/yv/ov are (T, 128, w) HBM views; tile t is one contiguous 128*w block.
    # DMA moves the whole tile at once; compute runs in cw-wide chunks so
    # ScalarE (silu) and VectorE (multiply) overlap inside the tile.
    P = 128
    xt = nl.ndarray((P, w), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=xv[t, :, 0:w])
    yt = nl.ndarray((P, w), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=yt, src=yv[t, :, 0:w])
    nch = w // cw
    for c in range(nch):
        c0 = c * cw
        # silu(x) in fp32 on ScalarE (fp16 input upcast, fp32 result tile)
        st = nl.ndarray((P, cw), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=st, op=nl.silu, data=xt[:, c0:c0 + cw])
        # fp32 multiply by y on VectorE, single downcast, written in place
        # into the y chunk (no dependency with the next chunk's silu)
        nisa.tensor_tensor(dst=yt[:, c0:c0 + cw], data1=st,
                           data2=yt[:, c0:c0 + cw], op=nl.multiply)
    nisa.dma_copy(dst=ov[t, :, 0:w], src=yt)


@nki.jit
def _swiglu_kernel(x, y, f, r, tail_split, cw):
    M, N = x.shape
    P = 128
    dt = x.dtype
    out = nl.ndarray((M, N), dtype=dt, buffer=nl.shared_hbm)

    E = M * N
    TB = E // (P * f)          # number of big contiguous tiles (128 x f each)
    fs = f // r                # sub-tile width used for the tail tile

    cwb = min(cw, f)
    cws = min(cw, fs)

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

    for k in range(per):
        t = t0 + k
        split = 0
        if k == per - 1:
            split = tail_split
        if split > 0:
            # last tile of this program: smaller pieces to shorten the drain
            for s in range(r):
                _gate_tile(xs, ys, osm, t * r + s, fs, cws, dt)
        else:
            _gate_tile(xb, yb, ob, t, f, cwb, dt)
    return out


def _choose_tile(wp):
    # deterministic: aim for >= ~32 tiles per program, width in [1024, 4096]
    target = wp // 32
    f = 1024
    while f < target and f < 4096:
        f *= 2
    return f


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

    f = _choose_tile(wp)
    # big tiles must evenly cover the tensor and split evenly over programs
    while f > 1 and E % (n_prog * _P * f) != 0:
        f //= 2
    if E % (_P * f) != 0:
        f = 1

    r = 1
    while r < 4 and (f // (r * 2)) >= 512:
        r *= 2

    TB = E // (_P * f)
    per = TB // n_prog if TB % n_prog == 0 else TB
    tail_split = 1 if (r > 1 and per >= 2) else 0

    cw = _CW
    # chunk width must divide the tile widths (all powers of two here)
    while cw > 1 and (f % min(cw, f) != 0 or (f // r) % min(cw, f // r) != 0):
        cw //= 2

    out = _swiglu_kernel(x, y, f, r, tail_split, cw)

    fs = f // r
    _LAST_CONFIG = {
        "layout": "contiguous tiles: view (T,128,f), program-halves",
        "partitions": _P,
        "programs": n_prog,
        "elems_per_partition_per_program": wp,
        "tile_free": f,
        "tiles_per_program": per,
        "tail_split": tail_split,
        "tail_subtiles": r if tail_split else 1,
        "tail_subtile_free": fs if tail_split else f,
        "compute_chunk": min(cw, f),
        "tail_compute_chunk": min(cw, fs) if tail_split else min(cw, f),
        "silu_engine": "scalar(activation silu, fp32 chunk out)",
        "mul_engine": "vector(tensor_tensor, fp32 math, fp16 in-place into y tile)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
