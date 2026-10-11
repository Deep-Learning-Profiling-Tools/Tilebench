import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (literals / deterministic functions of the shape)
# ----------------------------------------------------------------------------
SLOPE = 0.01            # fixed negative slope (literal constant)

# main path: 125-partition tiles (n is a multiple of 250 in all configured cases)
NP_MAIN = 125           # partitions used per tile
F_MAX = 4000            # max row length (fp16 elements) -> <= 1 MB per DMA
T_MIN = 12              # at least ~12 tiles per physical core
SPLIT_MIN_F = 1000      # split first/last tile only if rows are this long
SPLIT_MIN_G = 250       # minimum sub-tile row length

# fallback path (n not a multiple of 250): 128-row tiles with partial tail
P_FB = 128
F_CAP_FB = 8192
F_FLOOR_FB = 1024
TT_FB = 12

_LAST_CONFIG = {}


# ----------------------------------------------------------------------------
# Kernel helpers
# ----------------------------------------------------------------------------
def _leaky_tile(x, out, row0, NP, G):
    # Process the contiguous element block [row0*G, (row0+NP)*G) as NP rows of G.
    total = x.shape[0]
    x2 = x.reshape((total // G, G))
    o2 = out.reshape((total // G, G))
    xt = nl.ndarray((NP, G), dtype=x.dtype, buffer=nl.sbuf)
    yt = nl.ndarray((NP, G), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xt, src=x2[row0:row0 + NP, :])
    # ScalarE: t = 0.01 * x (fp32 internally, single rounding to fp16)
    nisa.activation(dst=yt, op=nl.copy, data=xt, scale=SLOPE)
    # VectorE: y = max(x, t)  (pure selection -> leaky relu)
    nisa.tensor_tensor(dst=yt, data1=xt, data2=yt, op=nl.maximum,
                       engine=nisa.engine.vector)
    nisa.dma_copy(dst=o2[row0:row0 + NP, :], src=yt)


@nki.jit
def leaky_relu_kernel(x, F, NP, SPLIT):
    N = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nprog = nl.num_programs()
    pid = nl.program_id(0)
    per_prog = N // nprog
    tile_elems = NP * F
    T = per_prog // tile_elems
    base = pid * per_prog
    G = F // SPLIT
    sub_elems = NP * G
    for j in range(T):
        off = base + j * tile_elems
        if j == 0:
            for s in range(SPLIT):
                _leaky_tile(x, out, (off + s * sub_elems) // G, NP, G)
        elif j == T - 1:
            for s in range(SPLIT):
                _leaky_tile(x, out, (off + s * sub_elems) // G, NP, G)
        else:
            _leaky_tile(x, out, off // F, NP, F)
    return out


@nki.jit
def leaky_relu_kernel_fb(x, F):
    N = x.shape[0]
    R = N // F
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    x2 = x.reshape((R, F))
    o2 = out.reshape((R, F))

    nprog = nl.num_programs()
    pid = nl.program_id(0)
    r_begin = (R * pid) // nprog
    r_end = (R * (pid + 1)) // nprog
    nrows = r_end - r_begin
    ntiles = (nrows + P_FB - 1) // P_FB

    for i in range(ntiles):
        r0 = r_begin + i * P_FB
        rows = min(P_FB, r_end - r0)
        xt = nl.ndarray((P_FB, F), dtype=x.dtype, buffer=nl.sbuf)
        yt = nl.ndarray((P_FB, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt[0:rows, :], src=x2[r0:r0 + rows, :])
        nisa.activation(dst=yt[0:rows, :], op=nl.copy, data=xt[0:rows, :],
                        scale=SLOPE)
        nisa.tensor_tensor(dst=yt[0:rows, :], data1=xt[0:rows, :],
                           data2=yt[0:rows, :], op=nl.maximum,
                           engine=nisa.engine.vector)
        nisa.dma_copy(dst=o2[r0:r0 + rows, :], src=yt[0:rows, :])
    return out


# ----------------------------------------------------------------------------
# Host-side deterministic configuration
# ----------------------------------------------------------------------------
def _choose_main(n):
    if n % (2 * NP_MAIN) != 0:
        return None
    M = n // (2 * NP_MAIN)          # elements per partition per core
    cap = min(F_MAX, M // T_MIN)
    if cap < 1:
        cap = 1
    best_any = 0
    best16 = 0
    f = cap
    while f >= 1:
        if M % f == 0:
            if best_any == 0:
                best_any = f
            if f % 16 == 0:
                best16 = f
                break
        f -= 1
    if best_any == 0:
        best_any = 1
    F = best_any
    if best16 > 0 and 4 * best16 >= 3 * best_any:
        F = best16
    split = 1
    if F >= SPLIT_MIN_F:
        for s in (4, 5, 8, 2, 3):
            if F % s == 0 and F // s >= SPLIT_MIN_G:
                split = s
                break
    return F, split, M


def _choose_F_fb(n):
    per_core = max(1, n // 2)
    target = per_core // (P_FB * TT_FB)
    fmax = min(F_CAP_FB, max(F_FLOOR_FB, target))
    fmax = max(1, min(fmax, n))
    f = (fmax // 32) * 32
    while f >= 256:
        if n % f == 0:
            return f
        f -= 32
    f = fmax
    while f > 1:
        if n % f == 0:
            return f
        f -= 1
    return 1


def run(input: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(input.shape[0])
    choice = _choose_main(n)
    if choice is not None:
        F, split, M = choice
        out = leaky_relu_kernel(input, F, NP_MAIN, split)
        _LAST_CONFIG = {
            "path": "main",
            "partitions": NP_MAIN,
            "F": F,
            "split_edge_tiles": split,
            "sub_F": F // split,
            "elems_per_partition_per_core": M,
            "tiles_per_core": M // F,
            "programs": 2,
            "compute": "scalarE mul(0.01) + vectorE max",
            "f_max": F_MAX,
            "t_min": T_MIN,
        }
        return out
    F = _choose_F_fb(n)
    out = leaky_relu_kernel_fb(input, F)
    R = n // F
    _LAST_CONFIG = {
        "path": "fallback",
        "F": F,
        "rows": R,
        "tile_partitions": P_FB,
        "programs": 2,
        "compute": "scalarE mul(0.01) + vectorE max",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
