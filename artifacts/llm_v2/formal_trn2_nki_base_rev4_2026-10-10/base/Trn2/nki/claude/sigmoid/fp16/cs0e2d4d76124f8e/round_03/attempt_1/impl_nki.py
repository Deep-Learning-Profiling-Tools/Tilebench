# Sigmoid (fp16) for AWS Trainium2, NKI 0.7.0b1, LNC=2.
#
# Mapping (round-1 structure, refined)
#   * The flat input X of N elements is viewed, at no cost, as a 2-D HBM
#     matrix (rows, cols) with row-major order. `rows` is the largest divisor
#     of N that is <= 128 and keeps every row 64-byte aligned
#     (cols % 32 == 0). For all configured N this is 125, with a row stride of
#     40000*k bytes. That stride is not a power of two, so concurrent row
#     streams spread over the HBM channels. The power-of-two partition stride
#     tried in round 2 was slower.
#   * The two physical cores (nl.program_id) split the column range at a
#     32-column (64 B) boundary. Each core walks its columns in tiles of
#     (rows x width). All tile boundaries are multiples of 32 columns, so
#     every DMA descriptor starts on a 64-byte boundary.
#   * Per-core schedule: nm "main" tiles (width <= 4096 columns, which is
#     8 KiB per partition and ~1 MB per DMA), followed by two ramp-down tiles
#     of ~1/2 and ~1/4 of the nominal width. The ramp-down shortens the
#     pipeline drain.
#   * Each tile is processed in one pass: one DMA load (HBM -> SBUF), one
#     in-place ScalarE sigmoid (the fp16 operand is upcast to fp32, the
#     sigmoid is evaluated in fp32, and the result is cast to fp16 on write),
#     and one DMA store (SBUF -> HBM). No intermediate goes to global memory.
#   * Eight SBUF buffers rotate per core. Loads are issued 4 tiles ahead of
#     compute.
#   * A tiny warm-up sigmoid on a memset tile triggers the ScalarE table load
#     while the first DMA loads are in flight. Its result is multiplied by 0.0
#     and used as the (exactly zero) bias of every tile activation, so it
#     cannot be dead-code-eliminated. sigmoid(x + 0) == sigmoid(x).
# The configuration is a fixed deterministic function of N (see _plan).
import torch

try:  # registers the "neuron" device (torch also autoloads it)
    import torch_neuronx  # noqa: F401
except Exception:  # pragma: no cover
    pass

import nki
import nki.language as nl
import nki.isa as nisa

_PMAX = 128        # max partitions per on-chip tile
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


def _prologue(x2, rows, cbeg, nm, base, ext, ta, tb, al,
              b0, b1, b2, b3, b4, b5, b6, b7):
    nt = _ntiles(nm, tb)
    pre = min(_PREFETCH, nt)
    for j in range(pre):
        cj = cbeg + _tstart(j, nm, base, ext, ta, al)
        wj = _twidth(j, nm, base, ext, ta, tb, al)
        tj = _pick(j, b0, b1, b2, b3, b4, b5, b6, b7)
        nisa.dma_copy(dst=tj[0:rows, 0:wj], src=x2[0:rows, cj:cj + wj])


def _main(x2, o2, rows, cbeg, nm, base, ext, ta, tb, al, zb,
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
            nisa.dma_copy(dst=tj[0:rows, 0:wj], src=x2[0:rows, cj:cj + wj])
        nisa.dma_copy(dst=o2[0:rows, ci:ci + wi], src=ti[0:rows, 0:wi])


def _warmup(rows):
    # Early sigmoid on a tiny tile (loads the ScalarE table during the first
    # DMAs). Returns an exactly-zero fp32 [rows, 1] bias tile that depends on it.
    wt = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=wt, value=0.0)
    ws = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ws, op=nl.sigmoid, data=wt)
    zb = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=zb, data=ws, op0=nl.multiply, operand0=0.0)
    return zb


@nki.jit
def _sigmoid_kernel(X, rows, cols, al, fbuf, cb1,
                    nm0, base0, ext0, ta0, tb0,
                    nm1, base1, ext1, ta1, tb1):
    n = X.shape[0]
    out = nl.ndarray((n,), dtype=X.dtype, buffer=nl.shared_hbm)
    x2 = X.reshape((rows, cols))
    o2 = out.reshape((rows, cols))

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
            _prologue(x2, rows, 0, nm0, base0, ext0, ta0, tb0, al,
                      b0, b1, b2, b3, b4, b5, b6, b7)
            zb = _warmup(rows)
            _main(x2, o2, rows, 0, nm0, base0, ext0, ta0, tb0, al, zb,
                  b0, b1, b2, b3, b4, b5, b6, b7)
        else:
            _prologue(x2, rows, cb1, nm1, base1, ext1, ta1, tb1, al,
                      b0, b1, b2, b3, b4, b5, b6, b7)
            zb = _warmup(rows)
            _main(x2, o2, rows, cb1, nm1, base1, ext1, ta1, tb1, al, zb,
                  b0, b1, b2, b3, b4, b5, b6, b7)
    else:
        # Single program: process both column ranges one after the other.
        _prologue(x2, rows, 0, nm0, base0, ext0, ta0, tb0, al,
                  b0, b1, b2, b3, b4, b5, b6, b7)
        zb = _warmup(rows)
        _main(x2, o2, rows, 0, nm0, base0, ext0, ta0, tb0, al, zb,
              b0, b1, b2, b3, b4, b5, b6, b7)
        _prologue(x2, rows, cb1, nm1, base1, ext1, ta1, tb1, al,
                  b0, b1, b2, b3, b4, b5, b6, b7)
        _main(x2, o2, rows, cb1, nm1, base1, ext1, ta1, tb1, al, zb,
              b0, b1, b2, b3, b4, b5, b6, b7)
    return out


def _largest_divisor(n, need_align):
    d = min(_PMAX, n)
    while d >= 1:
        if n % d == 0:
            if (not need_align) or ((n // d) % _ALIGN == 0):
                return d
        d -= 1
    return 0


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
    rp = _largest_divisor(n, False)
    ra = _largest_divisor(n, True)
    if ra > 0 and 2 * ra >= rp:
        rows, al = ra, _ALIGN
    else:
        rows, al = rp, 1
    cols = n // rows
    c_mid = ((cols // 2) // al) * al
    s0 = _core_sched(c_mid, al)
    s1 = _core_sched(cols - c_mid, al)
    fbuf = max(s0["wmax"], s1["wmax"], 1)
    return rows, cols, al, c_mid, s0, s1, fbuf


def run(X: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(N)
    rows, cols, al, c_mid, s0, s1, fbuf = _plan(n)
    out = _sigmoid_kernel(X, rows, cols, al, fbuf, c_mid,
                          s0["nm"], s0["base"], s0["ext"], s0["ta"], s0["tb"],
                          s1["nm"], s1["base"], s1["ext"], s1["ta"], s1["tb"])
    _LAST_CONFIG = {
        "rows": rows,
        "cols": cols,
        "align_cols": al,
        "split_col": c_mid,
        "buf_cols": fbuf,
        "core0": dict(s0),
        "core1": dict(s1),
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
