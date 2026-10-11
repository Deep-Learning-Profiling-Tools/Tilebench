import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_INV127 = 1.0 / 127.0
_NBUF = 4
_NPROG = 2
_UNIT = 512          # tile boundaries are multiples of 512 elements (per partition)
_FU_MAX = 16         # max main tile = 16 * 512 = 8192 elements per partition
_RAMP_MIN_U = 16     # use ramped schedule when T >= 16 * 512

_LAST_CONFIG = {}


def _pick4(i, a, b, c, d):
    r = d
    if i == 0:
        r = a
    if i == 1:
        r = b
    if i == 2:
        r = c
    return r


def _bound(i, U, ramp, n_tiles, h1, h2, mid, n_mid):
    # Element offset (per partition) of the start of tile i (i == n_tiles -> T).
    r = _UNIT * ((i * U) // n_tiles)
    if ramp == 1:
        if i == 0:
            r = 0
        elif i == 1:
            r = _UNIT * h1
        elif i <= n_mid + 2:
            r = _UNIT * (h1 + h2 + ((i - 2) * mid) // n_mid)
        else:
            r = _UNIT * U
    return r


def _dequant_tile(ot, xt, st, f0, f1, cols):
    # Elements [f0, f1) of the partition stream, split at row boundaries so each
    # piece uses a single row scale.  fp32 math, single rounding to fp16 at dst.
    p = f0
    while p < f1:
        g = p // cols
        q = min(f1, (g + 1) * cols)
        nisa.tensor_scalar(
            dst=ot[:, p - f0:q - f0],
            data=xt[:, p - f0:q - f0],
            op0=nl.multiply,
            operand0=st[:, g:g + 1],
            op1=nl.multiply,
            operand1=_INV127,
            engine=nisa.engine.vector,
        )
        p = q


@nki.jit
def _dequant_rowwise_kernel(x, state_x, n_buf):
    rows, cols = x.shape
    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    # Each program handles a contiguous block of rows; partition p owns G consecutive rows.
    G = rows // (n_prog * _P)
    T = G * cols
    U = T // _UNIT

    # ---- tile schedule (trace-time) ----
    ramp = 0
    n_tiles = U // 2
    fmax_u = 2
    h1 = 0
    h2 = 0
    mid = 0
    n_mid = 1
    if U >= _RAMP_MIN_U:
        ramp = 1
        Fu = min(_FU_MAX, U // 4)
        h1 = Fu // 4
        h2 = Fu // 2
        tl = Fu // 4
        mid = U - h1 - h2 - tl
        n_mid = (mid + Fu - 1) // Fu
        n_tiles = n_mid + 3
        fmax_u = max(h2, (mid + n_mid - 1) // n_mid)
    if n_tiles < 1:
        n_tiles = 1
        fmax_u = U
    fmax = _UNIT * fmax_u

    nb = min(n_buf, n_tiles)
    la = nb - 1

    out = nl.ndarray((rows, cols), dtype=nl.float16, buffer=nl.shared_hbm)

    xv = x.reshape((n_prog, _P, T))[pid]          # (128, T) int8
    ov = out.reshape((n_prog, _P, T))[pid]        # (128, T) fp16
    sv = state_x.reshape((n_prog, _P, G))[pid]    # (128, G) fp32

    # Row scales for this program (loaded once, reused for all elements of the row).
    st = nl.ndarray((_P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=st, src=sv)

    # Rotating buffers (explicit slots).
    xb0 = nl.ndarray((_P, fmax), dtype=nl.int8, buffer=nl.sbuf)
    xb1 = nl.ndarray((_P, fmax), dtype=nl.int8, buffer=nl.sbuf)
    xb2 = nl.ndarray((_P, fmax), dtype=nl.int8, buffer=nl.sbuf)
    xb3 = nl.ndarray((_P, fmax), dtype=nl.int8, buffer=nl.sbuf)
    ob0 = nl.ndarray((_P, fmax), dtype=nl.float16, buffer=nl.sbuf)
    ob1 = nl.ndarray((_P, fmax), dtype=nl.float16, buffer=nl.sbuf)
    ob2 = nl.ndarray((_P, fmax), dtype=nl.float16, buffer=nl.sbuf)
    ob3 = nl.ndarray((_P, fmax), dtype=nl.float16, buffer=nl.sbuf)

    for s in range(n_tiles + la):
        # ---- prefetch load of tile s ----
        if s < n_tiles:
            a0 = _bound(s, U, ramp, n_tiles, h1, h2, mid, n_mid)
            a1 = _bound(s + 1, U, ramp, n_tiles, h1, h2, mid, n_mid)
            xt_l = _pick4(s % nb, xb0, xb1, xb2, xb3)
            nisa.dma_copy(dst=xt_l[:, 0:a1 - a0], src=xv[:, a0:a1])

        # ---- compute + store of tile u ----
        u = s - la
        if u >= 0:
            f0 = _bound(u, U, ramp, n_tiles, h1, h2, mid, n_mid)
            f1 = _bound(u + 1, U, ramp, n_tiles, h1, h2, mid, n_mid)
            xt = _pick4(u % nb, xb0, xb1, xb2, xb3)
            ot = _pick4(u % nb, ob0, ob1, ob2, ob3)
            _dequant_tile(ot, xt, st, f0, f1, cols)
            nisa.dma_copy(dst=ov[:, f0:f1], src=ot[:, 0:f1 - f0])

    return out


def _make_config(rows, cols):
    G = rows // (_NPROG * _P)
    T = G * cols
    U = T // _UNIT
    ramp = 0
    n_tiles = U // 2
    fmax_u = 2
    h1 = 0
    h2 = 0
    mid = 0
    n_mid = 1
    Fu = 2
    if U >= _RAMP_MIN_U:
        ramp = 1
        Fu = min(_FU_MAX, U // 4)
        h1 = Fu // 4
        h2 = Fu // 2
        tl = Fu // 4
        mid = U - h1 - h2 - tl
        n_mid = (mid + Fu - 1) // Fu
        n_tiles = n_mid + 3
        fmax_u = max(h2, (mid + n_mid - 1) // n_mid)
    if n_tiles < 1:
        n_tiles = 1
        fmax_u = U
    sizes = []
    for i in range(n_tiles):
        b0 = _bound(i, U, ramp, n_tiles, h1, h2, mid, n_mid)
        b1 = _bound(i + 1, U, ramp, n_tiles, h1, h2, mid, n_mid)
        sizes.append(int(b1 - b0))
    return {
        "programs": _NPROG,
        "rows_per_partition_per_program": G,
        "elems_per_partition_per_program": T,
        "ramped_schedule": bool(ramp),
        "main_tile_elems": int(_UNIT * Fu) if ramp else 1024,
        "max_tile_elems": int(_UNIT * fmax_u),
        "n_tiles_per_program": int(n_tiles),
        "tile_sizes": sizes,
        "n_bufs": min(_NBUF, n_tiles),
        "engine": "vector",
        "stores_per_tile": 1,
        "layout": "contiguous_rows_per_partition",
    }


def run(x, state_x, **kwargs):
    global _LAST_CONFIG
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    cfg = _make_config(rows, cols)
    out = _dequant_rowwise_kernel[_NPROG](x, state_x, _NBUF)
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
