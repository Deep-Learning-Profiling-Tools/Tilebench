import torch
import nki
import nki.language as nl
import nki.isa as nisa

# ----------------------------------------------------------------------------
# Static configuration (literals / deterministic functions of the shape)
# ----------------------------------------------------------------------------
P = 128                      # partitions per tile
SLOPE = 0.01                 # fixed negative slope (literal constant)
F_CAP = 8192                 # max free size (fp16 elements) per tile row
F_FLOOR = 1024               # lower bound of the target free size
TARGET_TILES_PER_CORE = 12   # aim for ~12 row tiles per physical core

_LAST_CONFIG = {}


def _choose_F(n):
    """Deterministic choice of the row length F (must divide n)."""
    per_core = max(1, n // 2)
    target = per_core // (P * TARGET_TILES_PER_CORE)
    fmax = min(F_CAP, max(F_FLOOR, target))
    fmax = max(1, min(fmax, n))
    # Prefer multiples of 32 (>= 64-byte aligned fp16 rows) that divide n.
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


@nki.jit
def leaky_relu_kernel(x, F):
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
    ntiles = (nrows + P - 1) // P

    for i in range(ntiles):
        r0 = r_begin + i * P
        rows = min(P, r_end - r0)
        xt = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        yt = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        # load
        nisa.dma_copy(dst=xt[0:rows, :], src=x2[r0:r0 + rows, :])
        # ScalarE: t = 0.01 * x  (fp32 internally, single rounding to fp16)
        nisa.activation(dst=yt[0:rows, :], op=nl.copy, data=xt[0:rows, :],
                        scale=SLOPE)
        # VectorE: y = max(x, t)  (pure selection -> exact leaky relu)
        nisa.tensor_tensor(dst=yt[0:rows, :], data1=xt[0:rows, :],
                           data2=yt[0:rows, :], op=nl.maximum,
                           engine=nisa.engine.vector)
        # store
        nisa.dma_copy(dst=o2[r0:r0 + rows, :], src=yt[0:rows, :])
    return out


def run(input: torch.Tensor, N: int, **kwargs):
    global _LAST_CONFIG
    n = int(input.shape[0])
    F = _choose_F(n)
    out = leaky_relu_kernel(input, F)
    R = n // F
    _LAST_CONFIG = {
        "F": F,
        "rows": R,
        "tile_partitions": P,
        "programs": 2,
        "rows_per_program": [R // 2, R - R // 2],
        "compute": "scalarE mul(0.01) + vectorE max",
        "target_tiles_per_core": TARGET_TILES_PER_CORE,
        "f_cap": F_CAP,
        "f_floor": F_FLOOR,
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
