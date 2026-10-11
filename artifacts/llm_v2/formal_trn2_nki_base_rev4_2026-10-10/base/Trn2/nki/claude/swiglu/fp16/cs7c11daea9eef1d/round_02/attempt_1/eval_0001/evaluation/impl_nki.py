import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_LAST_CONFIG = {}


def _process_part(xv, yv, ov, part, widths, offsets, dt):
    # Software-pipelined fused gate over one (P, Wp) strip:
    # loads for tile i+2 are issued before compute/store of tile i.
    P = 128
    T = len(widths)

    w0 = widths[0]
    o0 = offsets[0]
    xa = nl.ndarray((P, w0), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=xa, src=xv[part, :, o0:o0 + w0])
    ya = nl.ndarray((P, w0), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=ya, src=yv[part, :, o0:o0 + w0])

    xb = xa
    yb = ya
    if T > 1:
        w1 = widths[1]
        o1 = offsets[1]
        xb = nl.ndarray((P, w1), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=xv[part, :, o1:o1 + w1])
        yb = nl.ndarray((P, w1), dtype=dt, buffer=nl.sbuf)
        nisa.dma_copy(dst=yb, src=yv[part, :, o1:o1 + w1])

    for i in range(T):
        xc = xa
        yc = ya
        if i + 2 < T:
            w2 = widths[i + 2]
            o2 = offsets[i + 2]
            xc = nl.ndarray((P, w2), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=xc, src=xv[part, :, o2:o2 + w2])
            yc = nl.ndarray((P, w2), dtype=dt, buffer=nl.sbuf)
            nisa.dma_copy(dst=yc, src=yv[part, :, o2:o2 + w2])

        w = widths[i]
        c0 = offsets[i]
        # silu(x) in fp32 on ScalarE (input upcast internally), kept as fp32 tile
        st = nl.ndarray((P, w), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=st, op=nl.silu, data=xa)
        # fp32 multiply with y on VectorE, single downcast to the output dtype
        nisa.tensor_tensor(dst=xa, data1=st, data2=ya, op=nl.multiply)
        nisa.dma_copy(dst=ov[part, :, c0:c0 + w], src=xa)

        xa = xb
        ya = yb
        xb = xc
        yb = yc


@nki.jit
def _swiglu_kernel(x, y, nsplit, widths, offsets):
    M, N = x.shape
    P = 128
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    W = (M * N) // P
    Wp = W // nsplit
    xv = x.reshape((nsplit, P, Wp))
    yv = y.reshape((nsplit, P, Wp))
    ov = out.reshape((nsplit, P, Wp))

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    if nsplit % n_prog == 0:
        per = nsplit // n_prog
        for j in range(per):
            _process_part(xv, yv, ov, pid * per + j, widths, offsets, x.dtype)
    else:
        # fallback: every program does all parts (identical writes, still correct)
        for part in range(nsplit):
            _process_part(xv, yv, ov, part, widths, offsets, x.dtype)
    return out


def _schedule(wp):
    # Deterministic tile-width schedule for wp elements per partition:
    # ramp up (small first tiles), steady fmax tiles, ramp down (small last tiles).
    fmax = 4096 if wp >= 32768 else 2048
    fmin = 512
    widths = []
    if wp < 3 * fmax:
        c = 0
        while c < wp:
            w = min(fmin, wp - c)
            widths.append(w)
            c += w
        return fmax, widths
    up = [fmin]
    w = fmin
    while w < fmax:
        up.append(w)
        w *= 2
    down = list(reversed(up))
    ramp_sum = sum(up)
    mid = wp - 2 * ramp_sum
    nfull = mid // fmax
    rem = mid - nfull * fmax
    widths = list(up)
    for _ in range(nfull):
        widths.append(fmax)
    if rem > 0:
        widths.append(rem)
    widths.extend(down)
    return fmax, widths


def run(x, y):
    global _LAST_CONFIG
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.dim() == 2, "expected 2-D inputs (M, N)"
    M, N = int(x.shape[0]), int(x.shape[1])
    E = M * N
    assert E % _P == 0, "M*N must be a multiple of 128"

    W = E // _P
    nsplit = 2 if W % 2 == 0 else 1
    wp = W // nsplit

    fmax, widths = _schedule(wp)
    offsets = []
    c = 0
    for w in widths:
        offsets.append(c)
        c += w
    assert c == wp

    out = _swiglu_kernel(x, y, nsplit, tuple(widths), tuple(offsets))

    _LAST_CONFIG = {
        "layout": "flat_view(nsplit,128,Wp), strided column tiles",
        "partitions": _P,
        "nsplit": nsplit,
        "elems_per_partition_per_program": wp,
        "tile_free_max": fmax,
        "tile_free_min": 512,
        "num_tiles_per_program": len(widths),
        "tile_widths_head": list(widths[:6]),
        "tile_widths_tail": list(widths[-4:]),
        "prefetch_depth": 2,
        "silu_engine": "scalar(activation silu, fp32 out)",
        "mul_engine": "vector(tensor_tensor, fp32 math, fp16 out)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
