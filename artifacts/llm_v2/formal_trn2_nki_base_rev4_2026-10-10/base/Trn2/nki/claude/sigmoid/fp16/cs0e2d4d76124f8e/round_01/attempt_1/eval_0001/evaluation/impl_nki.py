# Sigmoid (fp16) for AWS Trainium2, NKI 0.7.0b1, LNC=2.
#
# Mapping
#   * The flat input X of N elements is viewed, at no cost, as a 2-D HBM matrix
#     (rows, cols). `rows` is the largest divisor of N that is <= 128, so the
#     reshape is exact. Each partition row is one contiguous run of `cols` elements.
#     For the configured sizes, rows is 125 or 128.
#   * The two physical cores of the logical core (nl.program_id) split the
#     column range. Each core walks its columns in tiles of
#     (rows x tile_f) elements. The last tile may be partial.
#   * Each tile is processed in one pass: one DMA load (HBM -> SBUF), one
#     ScalarE sigmoid activation in place, and one DMA store (SBUF -> HBM).
#     ScalarE upcasts the fp16 operand to fp32, evaluates the sigmoid in fp32,
#     and casts the result back to fp16 on write. No intermediate is written
#     to global memory.
#   * Eight SBUF buffers rotate per core. Loads are issued 4 tiles ahead of
#     compute, so loads, activations and stores overlap.
# Configuration is a fixed deterministic function of N (see _plan).
import torch

try:  # registers the "neuron" device (torch also autoloads it)
    import torch_neuronx  # noqa: F401
except Exception:  # pragma: no cover
    pass

import nki
import nki.language as nl
import nki.isa as nisa

_PMAX = 128        # max partitions per on-chip tile
_F_MAX = 4096      # max free-axis columns per tile (fp16: 8 KiB / partition, ~1 MiB per DMA)
_MIN_TILES = 8     # aim for at least this many tiles per core (pipelining for small N)
_MIN_F = 512       # never shrink tiles below this many columns (1 KiB DMA segments)
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


@nki.jit
def _sigmoid_kernel(X, rows, cols, tile_f):
    n = X.shape[0]
    out = nl.ndarray((n,), dtype=X.dtype, buffer=nl.shared_hbm)
    x2 = X.reshape((rows, cols))
    o2 = out.reshape((rows, cols))

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    c_begin = (cols * pid) // n_prog
    c_end = (cols * (pid + 1)) // n_prog
    width = c_end - c_begin
    n_tiles = (width + tile_f - 1) // tile_f

    b0 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)
    b1 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)
    b2 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)
    b3 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)
    b4 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)
    b5 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)
    b6 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)
    b7 = nl.ndarray((rows, tile_f), dtype=X.dtype, buffer=nl.sbuf)

    # Prologue: prefetch the first tiles.
    pre = min(_PREFETCH, n_tiles)
    for j in range(pre):
        cj = c_begin + j * tile_f
        wj = min(tile_f, c_end - cj)
        tj = _pick(j, b0, b1, b2, b3, b4, b5, b6, b7)
        nisa.dma_copy(dst=tj[0:rows, 0:wj], src=x2[0:rows, cj:cj + wj])

    # Steady state: compute tile i, prefetch tile i + _PREFETCH, store tile i.
    for i in range(n_tiles):
        ci = c_begin + i * tile_f
        wi = min(tile_f, c_end - ci)
        ti = _pick(i, b0, b1, b2, b3, b4, b5, b6, b7)
        # ScalarE: fp16 operand upcast to fp32, sigmoid in fp32, cast to fp16 on write.
        nisa.activation(dst=ti[0:rows, 0:wi], op=nl.sigmoid, data=ti[0:rows, 0:wi])
        j = i + _PREFETCH
        if j < n_tiles:
            cj = c_begin + j * tile_f
            wj = min(tile_f, c_end - cj)
            tj = _pick(j, b0, b1, b2, b3, b4, b5, b6, b7)
            nisa.dma_copy(dst=tj[0:rows, 0:wj], src=x2[0:rows, cj:cj + wj])
        nisa.dma_copy(dst=o2[0:rows, ci:ci + wi], src=ti[0:rows, 0:wi])
    return out


def _plan(n):
    # rows: largest divisor of n that is <= 128 (exact 2-D view of the flat vector)
    rows = 1
    d = min(_PMAX, n)
    while d >= 1:
        if n % d == 0:
            rows = d
            break
        d -= 1
    cols = n // rows
    w = (cols + 1) // 2                      # columns of the larger per-core share
    nt = (w + _F_MAX - 1) // _F_MAX
    nt_small = min(_MIN_TILES, max(1, w // _MIN_F))
    if nt_small > nt:
        nt = nt_small
    if nt < 1:
        nt = 1
    tile_f = (w + nt - 1) // nt
    if tile_f < 1:
        tile_f = 1
    return rows, cols, tile_f, nt


def run(X: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(N)
    rows, cols, tile_f, nt = _plan(n)
    out = _sigmoid_kernel(X, rows, cols, tile_f)
    _LAST_CONFIG = {
        "rows": rows,
        "cols": cols,
        "tile_f": tile_f,
        "tiles_per_core": nt,
        "nbuf": _NBUF,
        "prefetch": _PREFETCH,
        "f_max": _F_MAX,
        "min_tiles": _MIN_TILES,
        "min_f": _MIN_F,
        "lnc": 2,
        "engine": "scalar_activation_sigmoid_fp32_internal",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
