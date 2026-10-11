import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
PREFETCH = 3     # number of tile loads kept in flight ahead of compute/store

_LAST_CONFIG = {}


@nki.jit
def mul2_kernel(x, F):
    # x: (n,) in HBM; n == num_programs * T * 128 * F (guaranteed by run()).
    n = x.shape[0]
    out = nl.ndarray((n,), dtype=x.dtype, buffer=nl.shared_hbm)
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    chunk = P * F
    NT = n // chunk
    T = NT // n_prog
    base = pid * T
    x3 = x.reshape((NT, P, F))
    o3 = out.reshape((NT, P, F))

    # Software pipeline: loads are issued (in program order) PREFETCH tiles
    # ahead of the compute/store of the current tile, each into its own tile.
    a = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=a, src=x3[base])
    b = a
    c = a
    if T > 1:
        b = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=b, src=x3[base + 1])
    if T > 2:
        c = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=c, src=x3[base + 2])

    for i in range(T):
        cur = a
        a = b
        b = c
        if i + 3 < T:
            c = nl.ndarray((P, F), dtype=x.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=c, src=x3[base + i + 3])
        nisa.tensor_scalar(dst=cur, data=cur, op0=nl.multiply, operand0=2.0,
                           engine=nisa.engine.vector)
        nisa.dma_copy(dst=o3[base + i], src=cur)
    return out


def _choose_tile(n):
    # per-program, per-partition column count (2 programs under LNC=2)
    if n % (2 * P) != 0:
        raise ValueError("mul2: n must be a multiple of 256 for this implementation")
    C = n // (2 * P)
    f_target = min(4096, max(1024, C // 8))
    if f_target > C:
        f_target = C
    F = f_target
    while F > 1 and C % F != 0:
        F -= 1
    return F, C


def run(x: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG
    n = x.numel()
    F, C = _choose_tile(n)
    tiles_per_prog = C // F
    _LAST_CONFIG = {
        "programs": 2,
        "partitions": P,
        "tile_free": F,
        "tiles_per_program": tiles_per_prog,
        "prefetch_depth": PREFETCH,
        "engine": "vector",
        "op": "tensor_scalar_multiply_2",
    }
    return mul2_kernel(x, F)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
