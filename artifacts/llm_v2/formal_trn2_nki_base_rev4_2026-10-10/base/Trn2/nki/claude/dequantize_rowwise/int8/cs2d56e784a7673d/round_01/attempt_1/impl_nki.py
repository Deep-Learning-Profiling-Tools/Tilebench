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


@nki.jit
def _dequant_rowwise_kernel(x, state_x, f_target):
    rows, cols = x.shape
    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    # Each program handles a contiguous block of rows; partition p owns G consecutive rows.
    G = rows // (n_prog * _P)
    T = G * cols
    m = cols // 512

    k = 1
    nc = 1
    if cols <= f_target:
        k = min(G, f_target // cols)
    else:
        nc = (cols + f_target - 1) // f_target
    n_rg = (G + k - 1) // k
    n_tiles = n_rg * nc
    nb = min(_NBUF, n_tiles)
    la = nb - 1

    out = nl.ndarray((rows, cols), dtype=nl.float16, buffer=nl.shared_hbm)

    xv = x.reshape((n_prog, _P, T))[pid]          # (128, T) int8
    ov = out.reshape((n_prog, _P, T))[pid]        # (128, T) fp16
    sv = state_x.reshape((n_prog, _P, G))[pid]    # (128, G) fp32

    # Row scales for this program (loaded once, reused for all elements of the row).
    st = nl.ndarray((_P, G), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=st, src=sv)

    # Rotating buffers (explicit slots).
    xb0 = nl.ndarray((_P, f_target), dtype=nl.int8, buffer=nl.sbuf)
    xb1 = nl.ndarray((_P, f_target), dtype=nl.int8, buffer=nl.sbuf)
    xb2 = nl.ndarray((_P, f_target), dtype=nl.int8, buffer=nl.sbuf)
    xb3 = nl.ndarray((_P, f_target), dtype=nl.int8, buffer=nl.sbuf)
    ob0 = nl.ndarray((_P, f_target), dtype=nl.float16, buffer=nl.sbuf)
    ob1 = nl.ndarray((_P, f_target), dtype=nl.float16, buffer=nl.sbuf)
    ob2 = nl.ndarray((_P, f_target), dtype=nl.float16, buffer=nl.sbuf)
    ob3 = nl.ndarray((_P, f_target), dtype=nl.float16, buffer=nl.sbuf)

    for s in range(n_tiles + la):
        # ---- prefetch load of tile s ----
        if s < n_tiles:
            rg_l = s // nc
            c_l = s % nc
            g0_l = rg_l * k
            kk_l = min(k, G - g0_l)
            c0_l = 512 * ((c_l * m) // nc)
            c1_l = 512 * (((c_l + 1) * m) // nc)
            f0_l = g0_l * cols + c0_l
            fs_l = (kk_l - 1) * cols + (c1_l - c0_l)
            xt_l = _pick4(s % nb, xb0, xb1, xb2, xb3)
            nisa.dma_copy(dst=xt_l[:, 0:fs_l], src=xv[:, f0_l:f0_l + fs_l])

        # ---- compute + store of tile u ----
        u = s - la
        if u >= 0:
            rg = u // nc
            c = u % nc
            g0 = rg * k
            kk = min(k, G - g0)
            c0 = 512 * ((c * m) // nc)
            c1 = 512 * (((c + 1) * m) // nc)
            w = c1 - c0
            f0 = g0 * cols + c0
            fs = (kk - 1) * cols + w
            xt = _pick4(u % nb, xb0, xb1, xb2, xb3)
            ot = _pick4(u % nb, ob0, ob1, ob2, ob3)
            for j in range(kk):
                a = j * cols
                g = g0 + j
                nisa.tensor_scalar(
                    dst=ot[:, a:a + w],
                    data=xt[:, a:a + w],
                    op0=nl.multiply,
                    operand0=st[:, g:g + 1],
                    op1=nl.multiply,
                    operand1=_INV127,
                    engine=nisa.engine.vector,
                )
            nisa.dma_copy(dst=ov[:, f0:f0 + fs], src=ot[:, 0:fs])

    return out


def _make_config(rows, cols):
    G = rows // (_NPROG * _P)
    T = G * cols
    f = (T // 8) // 512 * 512
    f = max(_FMIN, min(_FMAX, f))
    if cols <= f:
        k = min(G, f // cols)
        nc = 1
    else:
        k = 1
        nc = (cols + f - 1) // f
    n_tiles = ((G + k - 1) // k) * nc
    return {
        "programs": _NPROG,
        "rows_per_partition_per_program": G,
        "f_target": f,
        "rows_per_tile": k,
        "col_chunks_per_row": nc,
        "n_tiles_per_program": n_tiles,
        "n_bufs": min(_NBUF, n_tiles),
        "engine": "vector",
        "layout": "contiguous_rows_per_partition",
    }


def run(x, state_x, **kwargs):
    global _LAST_CONFIG
    rows = int(x.shape[0])
    cols = int(x.shape[1])
    cfg = _make_config(rows, cols)
    out = _dequant_rowwise_kernel[_NPROG](x, state_x, cfg["f_target"])
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
