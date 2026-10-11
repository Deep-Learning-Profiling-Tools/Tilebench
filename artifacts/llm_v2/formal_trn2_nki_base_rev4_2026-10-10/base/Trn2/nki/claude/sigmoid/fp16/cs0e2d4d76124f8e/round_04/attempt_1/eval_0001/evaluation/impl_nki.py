# Sigmoid (fp16) for AWS Trainium2, NKI 0.7.0b1, LNC=2.
#
# Mapping
#   * The flat input X (N elements) is split into a "bulk" part, viewed as a
#     (128, cs) row-major matrix with cs = 32*m (m odd, m <= N // 4096), and a
#     small tail (< 8192 elements, a multiple of 32 for all configured N).
#       - 128 rows: every DMA and every ScalarE instruction covers all 128
#         partitions.
#       - The row stride is cs*2 = 64*m bytes with m odd: 64-byte aligned
#         (every descriptor starts on a 64 B boundary) and with the smallest
#         possible power-of-two factor, so the concurrent row streams spread
#         across HBM channels. Power-of-two strides were measured slower.
#     Bulk windows are addressed with an explicit access pattern on the flat
#     HBM tensor: element (p, j) of a window = X[c0 + p*cs + j].
#   * The two physical cores (nl.program_id) split the bulk column range at a
#     32-column boundary. Each core walks its columns in (128 x width) tiles
#     (all tile boundaries are multiples of 32 columns): nm main tiles
#     (<= 4096 columns, ~1 MB per DMA) followed by two ramp-down tiles
#     (~1/2 and ~1/4 of the nominal width) that shorten the drain.
#   * The tail is cut into two contiguous (r, 32) pieces, one per core. Each
#     piece's load is issued first, and its sigmoid/store right after the
#     ScalarE warm-up, while the first big load is still in flight.
#   * Every tile is processed in one pass: one DMA load (HBM -> SBUF), one
#     in-place ScalarE sigmoid, and one DMA store (SBUF -> HBM).
#     ScalarE upcasts the fp16 operand to fp32, evaluates the sigmoid in fp32
#     and casts the result to fp16 on write. No intermediate goes to global
#     memory; every element is read once and written once.
#   * Eight SBUF buffers rotate per core; loads are issued 4 tiles ahead.
#   * A tiny warm-up sigmoid loads the ScalarE table during the first DMAs.
#     Its result times 0.0 (on VectorE) is the exactly-zero bias of every
#     activation, which keeps it alive: sigmoid(x + 0) == sigmoid(x).
# The configuration is a fixed deterministic function of N (see _plan).
import torch

try:  # registers the "neuron" device (torch also autoloads it)
    import torch_neuronx  # noqa: F401
except Exception:  # pragma: no cover
    pass

import nki
import nki.language as nl
import nki.isa as nisa

_PMAX = 128        # partitions per on-chip tile (bulk rows)
_ALIGN = 32        # column alignment of tile boundaries (32 fp16 = 64 B)
_F_MAX = 4096      # max columns of a main tile (multiple of _ALIGN)
_MIN_TILES = 8     # aim for at least this many tiles per core
_MIN_F = 512       # do not shrink nominal tiles below this many columns
_MIN_TAIL = 256    # smallest ramp-down tile width (columns); else no ramp-down
_NBUF = 8          # rotating SBUF buffers per core (must match _pick)
_PREFETCH = 4      # loads issued ahead of compute (must be < _NBUF)

_LAST_CONFIG = {}


def _pick(j, b0, b1, b2, b3, b4, b5, b6, b7):
    # Select rotating buffer j mod 8 (trace-time selection).
    s = j - (j // 8) * 8
    r = b0
    if s == 1:
        r = b1
    if s == 2:
        r = b2
    if s == 3:
        r = b3
    if s == 4:
        r = b4
    if s == 5:
        r = b5
    if s == 6:
        r = b6
    if s == 7:
        r = b7
    return r


def _tstart(i, nm, base, ext, ta, al):
    # Column offset (relative to the core's first column) of tile i.
    s = al * (i * base + min(i, ext))
    if i > nm:
        s = al * (nm * base + ext) + ta
    return s


def _twidth(i, nm, base, ext, ta, tb, al):
    # Column width of tile i: main tiles, then the two ramp-down tiles.
    w = al * base
    if i < ext:
        w = w + al
    if i == nm:
        w = ta
    if i > nm:
        w = tb
    return w


def _ntiles(nm, tb):
    nt = nm
    if tb > 0:
        nt = nm + 2
    return nt


def _win(t, stride, nrows, off, width):
    # (nrows x width) window of a flat HBM tensor: element (p, j) = t[off + p*stride + j].
    return t.ap(pattern=[[stride, nrows], [1, width]], offset=off)


def _prologue(X, rows, cs, cbeg, nm, base, ext, ta, tb, al,
              b0, b1, b2, b3, b4, b5, b6, b7):
    nt = _ntiles(nm, tb)
    pre = min(_PREFETCH, nt)
    for j in range(pre):
        cj = cbeg + _tstart(j, nm, base, ext, ta, al)
        wj = _twidth(j, nm, base, ext, ta, tb, al)
        tj = _pick(j, b0, b1, b2, b3, b4, b5, b6, b7)
        nisa.dma_copy(dst=tj[0:rows, 0:wj], src=_win(X, cs, rows, cj, wj))


def _main(X, out, rows, cs, cbeg, nm, base, ext, ta, tb, al, zb,
          b0, b1, b2, b3, b4, b5, b6, b7):
    nt = _ntiles(nm, tb)
    for i in range(nt):
        ci = cbeg + _tstart(i, nm, base, ext, ta, al)
        wi = _twidth(i, nm, base, ext, ta, tb, al)
        ti = _pick(i, b0, b1, b2, b3, b4, b5, b6, b7)
        # ScalarE: fp16 operand upcast to fp32, sigmoid(x + 0) in fp32, fp16 on write.
        nisa.activation(dst=ti[0:rows, 0:wi], op=nl.sigmoid,
                        data=ti[0:rows, 0:wi], bias=zb)
        j = i + _PREFETCH
        if j < nt:
            cj = cbeg + _tstart(j, nm, base, ext, ta, al)
            wj = _twidth(j, nm, base, ext, ta, tb, al)
            tj = _pick(j, b0, b1, b2, b3, b4, b5, b6, b7)
            nisa.dma_copy(dst=tj[0:rows, 0:wj], src=_win(X, cs, rows, cj, wj))
        nisa.dma_copy(dst=_win(out, cs, rows, ci, wi), src=ti[0:rows, 0:wi])


def _tail_load(X, q, off, r, c):
    # Contiguous tail piece of r*c elements at element offset `off` -> q[0:r, 0:c].
    if r > 0:
        nisa.dma_copy(dst=q[0:r, 0:c], src=_win(X, c, r, off, c))


def _tail_finish(out, q, off, r, c, zb):
    if r > 0:
        nisa.activation(dst=q[0:r, 0:c], op=nl.sigmoid,
                        data=q[0:r, 0:c], bias=zb[0:r, 0:1])
        nisa.dma_copy(dst=_win(out, c, r, off, c), src=q[0:r, 0:c])


def _tail_full(X, out, off, r, c, zb):
    # Self-contained piece (only used when N is not a multiple of 32).
    if r > 0:
        q = nl.ndarray((r, c), dtype=X.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q[0:r, 0:c], src=_win(X, c, r, off, c))
        nisa.activation(dst=q[0:r, 0:c], op=nl.sigmoid,
                        data=q[0:r, 0:c], bias=zb[0:r, 0:1])
        nisa.dma_copy(dst=_win(out, c, r, off, c), src=q[0:r, 0:c])


def _warmup(rows):
    # Early sigmoid on a tiny tile (loads the ScalarE table during the first
    # DMAs). Returns an exactly-zero fp32 [rows, 1] bias tile that depends on it.
    wt = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=wt, value=0.0, engine=nisa.engine.vector)
    ws = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ws, op=nl.sigmoid, data=wt)
    zb = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=zb, data=ws, op0=nl.multiply, operand0=0.0,
                       engine=nisa.engine.vector)
    return zb


@nki.jit
def _sigmoid_kernel(X, rows, cs, al, fbuf, cb1,
                    nm0, base0, ext0, ta0, tb0,
                    nm1, base1, ext1, ta1, tb1,
                    oa, ra, ob, rb, orem, nrem):
    n = X.shape[0]
    out = nl.ndarray((n,), dtype=X.dtype, buffer=nl.shared_hbm)

    b0 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)
    b1 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)
    b2 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)
    b3 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)
    b4 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)
    b5 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)
    b6 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)
    b7 = nl.ndarray((rows, fbuf), dtype=X.dtype, buffer=nl.sbuf)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    if n_prog > 1:
        if pid == 0:
            qa = nl.ndarray((_PMAX, _ALIGN), dtype=X.dtype, buffer=nl.sbuf)
            _tail_load(X, qa, oa, ra, al)
            _prologue(X, rows, cs, 0, nm0, base0, ext0, ta0, tb0, al,
                      b0, b1, b2, b3, b4, b5, b6, b7)
            zb = _warmup(rows)
            _tail_finish(out, qa, oa, ra, al, zb)
            _main(X, out, rows, cs, 0, nm0, base0, ext0, ta0, tb0, al, zb,
                  b0, b1, b2, b3, b4, b5, b6, b7)
        else:
            qb = nl.ndarray((_PMAX, _ALIGN), dtype=X.dtype, buffer=nl.sbuf)
            _tail_load(X, qb, ob, rb, al)
            _prologue(X, rows, cs, cb1, nm1, base1, ext1, ta1, tb1, al,
                      b0, b1, b2, b3, b4, b5, b6, b7)
            zb = _warmup(rows)
            _tail_finish(out, qb, ob, rb, al, zb)
            _main(X, out, rows, cs, cb1, nm1, base1, ext1, ta1, tb1, al, zb,
                  b0, b1, b2, b3, b4, b5, b6, b7)
            _tail_full(X, out, orem, 1 if nrem > 0 else 0, nrem, zb)
    else:
        # Single program: process both column ranges one after the other.
        qa = nl.ndarray((_PMAX, _ALIGN), dtype=X.dtype, buffer=nl.sbuf)
        qb = nl.ndarray((_PMAX, _ALIGN), dtype=X.dtype, buffer=nl.sbuf)
        _tail_load(X, qa, oa, ra, al)
        _tail_load(X, qb, ob, rb, al)
        _prologue(X, rows, cs, 0, nm0, base0, ext0, ta0, tb0, al,
                  b0, b1, b2, b3, b4, b5, b6, b7)
        zb = _warmup(rows)
        _tail_finish(out, qa, oa, ra, al, zb)
        _tail_finish(out, qb, ob, rb, al, zb)
        _main(X, out, rows, cs, 0, nm0, base0, ext0, ta0, tb0, al, zb,
              b0, b1, b2, b3, b4, b5, b6, b7)
        _prologue(X, rows, cs, cb1, nm1, base1, ext1, ta1, tb1, al,
                  b0, b1, b2, b3, b4, b5, b6, b7)
        _main(X, out, rows, cs, cb1, nm1, base1, ext1, ta1, tb1, al, zb,
              b0, b1, b2, b3, b4, b5, b6, b7)
        _tail_full(X, out, orem, 1 if nrem > 0 else 0, nrem, zb)
    return out


def _core_sched(W, al):
    # Deterministic per-core tile schedule for W columns (W multiple of al).
    if W <= 0:
        return {"width": 0, "nm": 0, "base": 0, "ext": 0, "ta": 0, "tb": 0, "wmax": 0}
    u = W // al
    nt = (W + _F_MAX - 1) // _F_MAX
    nts = min(_MIN_TILES, max(1, W // _MIN_F))
    nt = max(nt, nts, 1)
    fn = u // nt
    ta_u = fn // 2
    tb_u = fn // 4
    if nt < 3 or tb_u * al < _MIN_TAIL:
        ta_u = 0
        tb_u = 0
    main_u = u - ta_u - tb_u
    nm = (main_u * al + _F_MAX - 1) // _F_MAX
    nm_min = nt - 1 if tb_u > 0 else nt
    nm = max(nm, nm_min, 1)
    if nm > main_u:
        nm = main_u
    base = main_u // nm
    ext = main_u - base * nm
    wmax = (base + (1 if ext > 0 else 0)) * al
    return {"width": W, "nm": nm, "base": base, "ext": ext,
            "ta": ta_u * al, "tb": tb_u * al, "wmax": wmax}


def _plan(n):
    rows = _PMAX
    al = _ALIGN
    m = n // (rows * al)
    if m > 0 and m % 2 == 0:
        m -= 1                      # odd m -> row stride 64*m bytes (64*odd)
    cs = al * m                     # bulk row length = row stride (elements)
    bulk = rows * cs
    tail = n - bulk                 # < 8192 elements
    t = tail // al                  # full 32-element tail rows (<= 255)
    nrem = tail - t * al            # 0 for every configured N
    ra = (t + 1) // 2               # tail rows on core 0 (<= 128)
    rb = t - ra                     # tail rows on core 1 (<= 127)
    oa = bulk
    ob = bulk + ra * al
    orem = bulk + t * al
    c_mid = al * (m // 2)
    s0 = _core_sched(c_mid, al)
    s1 = _core_sched(cs - c_mid, al)
    fbuf = max(s0["wmax"], s1["wmax"], 1)
    return {"rows": rows, "al": al, "m": m, "cs": cs, "bulk": bulk,
            "c_mid": c_mid, "s0": s0, "s1": s1, "fbuf": fbuf,
            "oa": oa, "ra": ra, "ob": ob, "rb": rb,
            "orem": orem, "nrem": nrem}


def run(X: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(N)
    p = _plan(n)
    s0 = p["s0"]
    s1 = p["s1"]
    out = _sigmoid_kernel(X, p["rows"], p["cs"], p["al"], p["fbuf"], p["c_mid"],
                          s0["nm"], s0["base"], s0["ext"], s0["ta"], s0["tb"],
                          s1["nm"], s1["base"], s1["ext"], s1["ta"], s1["tb"],
                          p["oa"], p["ra"], p["ob"], p["rb"],
                          p["orem"], p["nrem"])
    _LAST_CONFIG = {
        "rows": p["rows"],
        "row_stride_cols": p["cs"],
        "row_stride_units64B": p["m"],
        "bulk_elems": p["bulk"],
        "align_cols": p["al"],
        "split_col": p["c_mid"],
        "buf_cols": p["fbuf"],
        "core0": dict(s0),
        "core1": dict(s1),
        "tail_rows_core0": p["ra"],
        "tail_rows_core1": p["rb"],
        "tail_cols": p["al"],
        "rem_elems": p["nrem"],
        "nbuf": _NBUF,
        "prefetch": _PREFETCH,
        "f_max": _F_MAX,
        "min_tiles": _MIN_TILES,
        "min_f": _MIN_F,
        "min_tail": _MIN_TAIL,
        "lnc": 2,
        "engine": "scalar_activation_sigmoid_fp32_internal_zero_bias_warmup",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
