import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_INV127 = 1.0 / 127.0
_NBUF = 4
_FMIN = 1024
_FMAX = 4096
_NPROG = 2

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


def _dequant_vector(dst, src, st, base, a, b, cols):
    # Elements [a, b) of the tile (tile starts at per-partition offset `base`),
    # split at row boundaries so each piece uses one row scale.
    p = a
    while p < b:
        g = (base + p) // cols
        q = min(b, (g + 1) * cols - base)
        nisa.tensor_scalar(
            dst=dst[:, p - a:q - a],
            data=src[:, p:q],
            op0=nl.multiply,
            operand0=st[:, g:g + 1],
            op1=nl.multiply,
            operand1=_INV127,
            engine=nisa.engine.vector,
        )
        p = q


def _dequant_scalar(dst, src, s2, base, a, b, cols):
    p = a
    while p < b:
        g = (base + p) // cols
        q = min(b, (g + 1) * cols - base)
        nisa.activation(
            dst=dst[:, p - a:q - a],
            op=nl.copy,
            data=src[:, p:q],
            scale=s2[:, g:g + 1],
        )
        p = q


@nki.jit
def _dequant_rowwise_kernel(x, state_x, f_t, n_buf):
    rows, cols = x.shape
    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    # Each program handles a contiguous block of rows; partition p owns G consecutive rows.
    G = rows // (n_prog * _P)
    T = G * cols
    n_tiles = (T + f_t - 1) // f_t
    nb = min(n_buf, n_tiles)
    la = nb - 1
    hv = f_t // 2
    hs = f_t - hv

    out = nl.ndarray((rows, cols), dtype=nl.float16, buffer=nl.shared_hbm)

    xv = x.reshape((n_prog, _P, T))[pid]          # (128, T) int8
    ov = out.reshape((n_prog, _P, T))[pid]        # (128, T) fp16
    sv = state_x.reshape((n_prog, _P, G))[pid]    # (128, G) fp32

    # Row scales (loaded once per program, reused for every element of the row).
    st = nl.ndarray((_P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=st, src=sv)
    # fp32 scale * fp32 (1/127) for the ScalarE half.
    s2 = nl.ndarray((_P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=s2, op=nl.copy, data=st, scale=_INV127)

    # Rotating buffers (explicit slots).
    xb0 = nl.ndarray((_P, f_t), dtype=nl.int8, buffer=nl.sbuf)
    xb1 = nl.ndarray((_P, f_t), dtype=nl.int8, buffer=nl.sbuf)
    xb2 = nl.ndarray((_P, f_t), dtype=nl.int8, buffer=nl.sbuf)
    xb3 = nl.ndarray((_P, f_t), dtype=nl.int8, buffer=nl.sbuf)
    vb0 = nl.ndarray((_P, hv), dtype=nl.float16, buffer=nl.sbuf)
    vb1 = nl.ndarray((_P, hv), dtype=nl.float16, buffer=nl.sbuf)
    vb2 = nl.ndarray((_P, hv), dtype=nl.float16, buffer=nl.sbuf)
    vb3 = nl.ndarray((_P, hv), dtype=nl.float16, buffer=nl.sbuf)
    sb0 = nl.ndarray((_P, hs), dtype=nl.float16, buffer=nl.sbuf)
    sb1 = nl.ndarray((_P, hs), dtype=nl.float16, buffer=nl.sbuf)
    sb2 = nl.ndarray((_P, hs), dtype=nl.float16, buffer=nl.sbuf)
    sb3 = nl.ndarray((_P, hs), dtype=nl.float16, buffer=nl.sbuf)

    for s in range(n_tiles + la):
        # ---- prefetch load of tile s ----
        if s < n_tiles:
            f0l = s * f_t
            fsl = min(f_t, T - f0l)
            xtl = _pick4(s % nb, xb0, xb1, xb2, xb3)
            nisa.dma_copy(dst=xtl[:, 0:fsl], src=xv[:, f0l:f0l + fsl])

        # ---- compute (VectorE + ScalarE halves) + stores of tile u ----
        u = s - la
        if u >= 0:
            f0 = u * f_t
            fs = min(f_t, T - f0)
            h = fs // 2
            slot = u % nb
            xt = _pick4(slot, xb0, xb1, xb2, xb3)
            vt = _pick4(slot, vb0, vb1, vb2, vb3)
            sc = _pick4(slot, sb0, sb1, sb2, sb3)
            _dequant_vector(vt, xt, st, f0, 0, h, cols)
            _dequant_scalar(sc, xt, s2, f0, h, fs, cols)
            nisa.dma_copy(dst=ov[:, f0:f0 + h], src=vt[:, 0:h])
            nisa.dma_copy(dst=ov[:, f0 + h:f0 + fs], src=sc[:, 0:fs - h])

    return out


def _make_config(rows, cols):
    G = rows // (_NPROG * _P)
    T = G * cols
    f = (T // 8) // 512 * 512
    f = max(_FMIN, min(_FMAX, f))
    n_tiles = (T + f - 1) // f
    return {
        "programs": _NPROG,
        "rows_per_partition_per_program": G,
        "f_target": f,
        "n_tiles_per_program": n_tiles,
        "n_bufs": min(_NBUF, n_tiles),
        "engines": "vector+scalar",
        "vector_share": 0.5,
        "stores_per_tile": 2,
        "layout": "contiguous_rows_per_partition",
    }


def run(x, state_x, **kwargs):
    global _LAST_CONFIG
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    cfg = _make_config(rows, cols)
    out = _dequant_rowwise_kernel[_NPROG](x, state_x, cfg["f_target"], _NBUF)
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
