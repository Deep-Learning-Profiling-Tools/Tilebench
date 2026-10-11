# Sigmoid (fp16) for AWS Trainium2, NKI 0.7.0b1, LNC=2.
#
# Mapping
#   * The two physical cores of the logical core each take one contiguous half
#     of the flat vector. The split point is rounded down to a 2048-element
#     boundary, so core 1 starts on a 4 KiB boundary.
#   * Each core walks its range in contiguous tiles of 128 partitions x F
#     elements. Partition p of a tile holds F consecutive elements. F is a
#     multiple of 128 elements, so every per-partition chunk is >= 256 B
#     aligned.
#   * Tile sizes: most tiles have width F1. The last 2-3 tiles have width
#     F2 = F1/2, which shortens the pipeline tail. The final remainder
#     (< 128*F2 elements) is one block of whole rows (width w3, a multiple of
#     64) plus a single-partition tail tile. Edge handling is exact: no element
#     at index >= N is read or written.
#   * Per tile, in one pass: DMA load (HBM -> SBUF), in-place ScalarE sigmoid
#     (the fp16 operand is upcast to fp32, the sigmoid is evaluated in fp32,
#     and the result is cast to fp16 on write), DMA store (SBUF -> HBM).
#     No intermediate is written to global memory.
#   * 12 rotating SBUF buffers per core. Loads are issued 6 tiles ahead.
#     Full-tile DMAs are split into two 64-partition transfers so more DMA
#     queues work in parallel.
# The configuration is a fixed deterministic function of N (see _plan).
import torch

try:  # registers the "neuron" device (torch also autoloads it)
    import torch_neuronx  # noqa: F401
except Exception:  # pragma: no cover
    pass

import nki
import nki.language as nl
import nki.isa as nisa

_P = 128            # partitions per tile
_F_MAX = 4096       # max width (elements per partition) of a main tile (1 MiB fp16 tiles)
_F_MIN = 512        # min width of a main tile
_F_GRAN = 256       # main-tile width granularity (elements)
_MIN_TILES = 8      # aim for at least this many main tiles per core
_SPLIT_ALIGN = 2048 # core split point alignment (elements)
_TAIL_ALIGN = 64    # remainder block width granularity (elements)
_NBUF = 12          # rotating SBUF buffers per core (must match _pick)
_PREFETCH = 6       # loads issued ahead of compute (< _NBUF)
_DSPLIT = 2         # DMA transfers per full 128-partition tile

_LAST_CONFIG = {}


def _pick(j, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11):
    # Select rotating buffer j mod 12 (trace-time selection).
    s = j - (j // 12) * 12
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
    if s == 8:
        r = b8
    if s == 9:
        r = b9
    if s == 10:
        r = b10
    if s == 11:
        r = b11
    return r


def _pstart(j, base, n1, f1, f2):
    # Element offset of piece j.
    s = base + j * (_P * f1)
    if j >= n1:
        s = base + n1 * (_P * f1) + (j - n1) * (_P * f2)
    return s


def _prows(j, n12, rows3):
    r = _P
    if j >= n12:
        r = rows3
    return r


def _pwidth(j, n1, n12, f1, f2, w3):
    w = f1
    if j >= n1:
        w = f2
    if j >= n12:
        w = w3
    return w


def _load_piece(x2, buf, tt, j, base, n1, n12, f1, f2, rows3, w3, t3):
    start = _pstart(j, base, n1, f1, f2)
    rows = _prows(j, n12, rows3)
    width = _pwidth(j, n1, n12, f1, f2, w3)
    if rows == _P:
        h = _P // _DSPLIT
        for q in range(_DSPLIT):
            p0 = q * h
            nisa.dma_copy(
                dst=buf[p0:p0 + h, 0:width],
                src=x2.ap(pattern=[[width, h], [1, width]], offset=start + p0 * width),
            )
    else:
        if rows > 0:
            nisa.dma_copy(
                dst=buf[0:rows, 0:width],
                src=x2.ap(pattern=[[width, rows], [1, width]], offset=start),
            )
    if j >= n12:
        if t3 > 0:
            ts = start + rows * width
            nisa.dma_copy(dst=tt[0:1, 0:t3], src=x2[0:1, ts:ts + t3])


def _compute_piece(buf, tt, j, n1, n12, f1, f2, rows3, w3, t3):
    rows = _prows(j, n12, rows3)
    width = _pwidth(j, n1, n12, f1, f2, w3)
    if rows > 0:
        # ScalarE: fp16 operand upcast to fp32, sigmoid in fp32, cast to fp16 on write.
        nisa.activation(dst=buf[0:rows, 0:width], op=nl.sigmoid, data=buf[0:rows, 0:width])
    if j >= n12:
        if t3 > 0:
            nisa.activation(dst=tt[0:1, 0:t3], op=nl.sigmoid, data=tt[0:1, 0:t3])


def _store_piece(o2, buf, tt, j, base, n1, n12, f1, f2, rows3, w3, t3):
    start = _pstart(j, base, n1, f1, f2)
    rows = _prows(j, n12, rows3)
    width = _pwidth(j, n1, n12, f1, f2, w3)
    if rows == _P:
        h = _P // _DSPLIT
        for q in range(_DSPLIT):
            p0 = q * h
            nisa.dma_copy(
                dst=o2.ap(pattern=[[width, h], [1, width]], offset=start + p0 * width),
                src=buf[p0:p0 + h, 0:width],
            )
    else:
        if rows > 0:
            nisa.dma_copy(
                dst=o2.ap(pattern=[[width, rows], [1, width]], offset=start),
                src=buf[0:rows, 0:width],
            )
    if j >= n12:
        if t3 > 0:
            ts = start + rows * width
            nisa.dma_copy(dst=o2[0:1, ts:ts + t3], src=tt[0:1, 0:t3])


@nki.jit
def _sigmoid_kernel(X, n, split, f1, f2):
    total = X.shape[0]
    out = nl.ndarray((total,), dtype=X.dtype, buffer=nl.shared_hbm)
    x2 = X.reshape((1, total))
    o2 = out.reshape((1, total))

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    base = 0
    length = n
    if n_prog > 1:
        if pid == 0:
            length = split
        else:
            base = split
            length = n - split

    # Per-core piece plan (trace-time integers).
    blk2 = _P * f2
    m = length // blk2
    r = length - m * blk2
    n1 = 0
    if m >= 2:
        n1 = (m - 2) // 2
    n2 = m - 2 * n1
    n12 = n1 + n2
    w3 = 0
    rows3 = 0
    t3 = 0
    n_pieces = n12
    if r > 0:
        w3 = (((r + _P - 1) // _P + _TAIL_ALIGN - 1) // _TAIL_ALIGN) * _TAIL_ALIGN
        rows3 = r // w3
        t3 = r - rows3 * w3
        n_pieces = n12 + 1

    b0 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b1 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b2 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b3 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b4 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b5 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b6 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b7 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b8 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b9 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b10 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    b11 = nl.ndarray((_P, f1), dtype=X.dtype, buffer=nl.sbuf)
    tt = nl.ndarray((1, f1), dtype=X.dtype, buffer=nl.sbuf)

    # Prologue: prefetch the first pieces.
    pre = min(_PREFETCH, n_pieces)
    for j in range(pre):
        bj = _pick(j, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11)
        _load_piece(x2, bj, tt, j, base, n1, n12, f1, f2, rows3, w3, t3)

    # Steady state: compute piece i, prefetch piece i + _PREFETCH, store piece i.
    for i in range(n_pieces):
        bi = _pick(i, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11)
        _compute_piece(bi, tt, i, n1, n12, f1, f2, rows3, w3, t3)
        j = i + _PREFETCH
        if j < n_pieces:
            bj = _pick(j, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11)
            _load_piece(x2, bj, tt, j, base, n1, n12, f1, f2, rows3, w3, t3)
        _store_piece(o2, bi, tt, i, base, n1, n12, f1, f2, rows3, w3, t3)
    return out


def _plan(n):
    half = n // 2
    f = half // (_P * _MIN_TILES)
    f = (f // _F_GRAN) * _F_GRAN
    if f > _F_MAX:
        f = _F_MAX
    if f < _F_MIN:
        f = _F_MIN
    f1 = f
    f2 = f1 // 2
    split = (half // _SPLIT_ALIGN) * _SPLIT_ALIGN
    return f1, f2, split


def _core_plan(length, f1, f2):
    blk2 = _P * f2
    m = length // blk2
    r = length - m * blk2
    n1 = 0
    if m >= 2:
        n1 = (m - 2) // 2
    n2 = m - 2 * n1
    w3 = 0
    rows3 = 0
    t3 = 0
    if r > 0:
        w3 = (((r + _P - 1) // _P + _TAIL_ALIGN - 1) // _TAIL_ALIGN) * _TAIL_ALIGN
        rows3 = r // w3
        t3 = r - rows3 * w3
    return {
        "length": length,
        "tiles_f1": n1,
        "tiles_f2": n2,
        "rem_rows": rows3,
        "rem_width": w3,
        "rem_tail": t3,
    }


def run(X: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    total = int(X.shape[0])
    n = int(N)
    if n > total:
        n = total
    f1, f2, split = _plan(n)
    out = _sigmoid_kernel(X, n, split, f1, f2)
    _LAST_CONFIG = {
        "f1": f1,
        "f2": f2,
        "split": split,
        "core0": _core_plan(split, f1, f2),
        "core1": _core_plan(n - split, f1, f2),
        "nbuf": _NBUF,
        "prefetch": _PREFETCH,
        "dma_split": _DSPLIT,
        "f_max": _F_MAX,
        "min_tiles": _MIN_TILES,
        "lnc": 2,
        "engine": "scalar_activation_sigmoid_fp32_internal",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
