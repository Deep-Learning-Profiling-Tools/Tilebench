import nki
import nki.language as nl
import nki.isa as nisa

PMAX = 128
_UNIT = 1 << 19            # 128 * 4096 elements
_last_config = {}


@nki.jit
def vector_add_dmac(x, y, F):
    # Elementwise add performed by the DMA engines (fp32 internal math, cast to fp16 on write),
    # streaming HBM -> HBM. Each program (physical core) handles one contiguous half of the data,
    # split into (128, F) chunks issued as independent DMA-compute instructions.
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    Tt = n // (PMAX * F)
    T = Tt // n_prog
    t0 = pid * T

    x3 = x.reshape((Tt, PMAX, F))
    y3 = y.reshape((Tt, PMAX, F))
    o3 = out.reshape((Tt, PMAX, F))

    for t in range(T):
        nisa.dma_compute(dst=o3[t0 + t, :, :],
                         srcs=[x3[t0 + t, :, :], y3[t0 + t, :, :]],
                         reduce_op=nl.add)
    return out


@nki.jit
def vector_add_generic(x, y, F):
    # Fallback for lengths outside the configured domain (arbitrary n); partial tiles via slices.
    n = x.shape[0]
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    q = n // F
    r = n - q * F
    rr = 0
    while rr < q:
        rows = min(PMAX, q - rr)
        off = rr * F
        a = nl.ndarray((PMAX, F), dtype=x.dtype, buffer=nl.sbuf)
        b = nl.ndarray((PMAX, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a[0:rows, :], src=x.ap(pattern=[[F, rows], [1, F]], offset=off))
        nisa.dma_copy(dst=b[0:rows, :], src=y.ap(pattern=[[F, rows], [1, F]], offset=off))
        nisa.tensor_tensor(dst=a[0:rows, :], data1=a[0:rows, :], data2=b[0:rows, :], op=nl.add)
        nisa.dma_copy(dst=out.ap(pattern=[[F, rows], [1, F]], offset=off), src=a[0:rows, :])
        rr = rr + rows
    if r > 0:
        off = q * F
        a2 = nl.ndarray((1, F), dtype=x.dtype, buffer=nl.sbuf)
        b2 = nl.ndarray((1, F), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a2[0:1, 0:r], src=x.ap(pattern=[[r, 1], [1, r]], offset=off))
        nisa.dma_copy(dst=b2[0:1, 0:r], src=y.ap(pattern=[[r, 1], [1, r]], offset=off))
        nisa.tensor_tensor(dst=a2[0:1, 0:r], data1=a2[0:1, 0:r], data2=b2[0:1, 0:r], op=nl.add)
        nisa.dma_copy(dst=out.ap(pattern=[[r, 1], [1, r]], offset=off), src=a2[0:1, 0:r])
    return out


def _choose_config(n):
    n_prog = 2
    if n <= 0 or n % (n_prog * _UNIT) != 0:
        return {"path": "generic", "F": 2048}
    k = n // (n_prog * _UNIT)           # per-program elements = k * 128 * 4096
    # aim for >= 16 chunks per program, rows of 1..8 KiB
    target = 256 * k
    F = 512
    while F * 2 <= target and F < 4096:
        F *= 2
    T = n // (n_prog * PMAX * F)
    return {
        "path": "dma_compute_hbm",
        "programs": n_prog,
        "F": F,
        "chunk_elems": PMAX * F,
        "chunks_per_program": T,
    }


def run(x, y):
    global _last_config
    n = x.shape[0]
    cfg = _choose_config(n)
    if cfg["path"] == "dma_compute_hbm":
        out = vector_add_dmac(x, y, cfg["F"])
    else:
        out = vector_add_generic(x, y, cfg["F"])
    _last_config = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_last_config)
