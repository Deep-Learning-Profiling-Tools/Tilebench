import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (literals / deterministic functions of the shape)
# ----------------------------------------------------------------------------
P = 128                      # partitions per tile
SLOPE = 0.01                 # fixed negative slope (literal constant)
F_CAP = 8192                 # max row length (fp16 elements) per tile row
F_FLOOR = 1024               # lower bound of the target row length
TARGET_TILES_PER_CORE = 12   # aim for ~12 row tiles per physical core
CHUNK_MIN = 800              # min column width of a tail chunk (elements)
CHUNK_MAX_N = 8              # max number of tail chunks of the last tile

_LAST_CONFIG = {}


def _choose_F(n):
    """Deterministic choice of the row length F (must divide n)."""
    per_core = max(1, n // 2)
    target = per_core // (P * TARGET_TILES_PER_CORE)
    fmax = min(F_CAP, max(F_FLOOR, target))
    fmax = max(1, min(fmax, n))
    # Prefer multiples of 32 (64-byte aligned fp16 rows) that divide n.
    f = (fmax // 32) * 32
    while f >= 256:
        if n % f == 0:
            return f
        f -= 32
    # Fallback: largest divisor of n not exceeding fmax.
    f = fmax
    while f > 1:
        if n % f == 0:
            return f
        f -= 1
    return 1


def _choose_chunks(F):
    """Number of column chunks for the last full tile and the base width."""
    nch = F // CHUNK_MIN
    if nch > CHUNK_MAX_N:
        nch = CHUNK_MAX_N
    if nch < 1:
        nch = 1
    cw = F
    if nch > 1:
        cw = ((F // nch) // 32) * 32
        if cw < 32:
            nch = 1
            cw = F
    return nch, cw


# ----------------------------------------------------------------------------
# Kernel
# ----------------------------------------------------------------------------
def _leaky_block(x2, o2, r0, rows, c0, w, dt):
    # rows [r0, r0+rows), columns [c0, c0+w) of the (R, F) view
    xt = nl.ndarray((rows, w), dtype=dt, buffer=nl.sbuf)
    tt = nl.ndarray((rows, w), dtype=dt, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x2[r0:r0 + rows, c0:c0 + w])
    # ScalarE: t = 0.01 * x (fp32 internally, single rounding to fp16)
    nisa.activation(dst=tt, op=nl.copy, data=xt, scale=SLOPE)
    # VectorE: y = max(x, t) written in place (pure selection -> leaky relu)
    nisa.tensor_tensor(dst=xt, data1=xt, data2=tt, op=nl.maximum,
                       engine=nisa.engine.vector)
    nisa.dma_copy(dst=o2[r0:r0 + rows, c0:c0 + w], src=xt)


@nki.jit
def leaky_relu_kernel(x, F, NCH, CW):
    N = x.shape[0]
    R = N // F
    dt = x.dtype
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x2 = x.reshape((R, F))
    o2 = out.reshape((R, F))

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    r_begin = (R * pid) // nprog
    r_end = (R * (pid + 1)) // nprog
    nrows = r_end - r_begin
    nfull = nrows // P
    rem = nrows - nfull * P

    # Partial-row tile first: its compute overlaps the loads of full tiles.
    if rem > 0:
        _leaky_block(x2, o2, r_begin, rem, 0, F, dt)

    rf = r_begin + rem
    for i in range(nfull):
        r0 = rf + i * P
        if i == nfull - 1:
            # Last full tile in column chunks -> short pipeline drain.
            for c in range(NCH):
                if c < NCH - 1:
                    _leaky_block(x2, o2, r0, P, c * CW, CW, dt)
                else:
                    _leaky_block(x2, o2, r0, P, c * CW, F - c * CW, dt)
        else:
            _leaky_block(x2, o2, r0, P, 0, F, dt)
    return out


# ----------------------------------------------------------------------------
# Host entry point
# ----------------------------------------------------------------------------
def run(input: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(input.shape[0])
    F = _choose_F(n)
    nch, cw = _choose_chunks(F)
    out = leaky_relu_kernel(input, F, nch, cw)
    R = n // F
    r0 = R // 2
    r1 = R - r0
    _LAST_CONFIG = {
        "F": F,
        "rows": R,
        "tile_partitions": P,
        "programs": 2,
        "rows_per_program": [r0, r1],
        "full_tiles_per_program": [r0 // P, r1 // P],
        "partial_rows_per_program": [r0 % P, r1 % P],
        "partial_tile_position": "first",
        "tail_chunks": nch,
        "tail_chunk_width": cw,
        "compute": "scalarE mul(0.01) + vectorE max (in place)",
        "target_tiles_per_core": TARGET_TILES_PER_CORE,
        "f_cap": F_CAP,
        "f_floor": F_FLOOR,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
