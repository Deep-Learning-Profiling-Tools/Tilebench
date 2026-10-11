import torch
import nki
import nki.language as nl
import nki.isa as nisa

P = 128
MAXW = 8192      # max columns per main tile (per partition) -> 2 MB fp16 DMA per core
MINW = 512       # min columns per tile piece (1 KB contiguous per partition row)

_LAST_CONFIG = {}


@nki.jit
def mul2_kernel(x, C, widths):
    # x: (n,) in HBM; n == num_programs * 128 * C (guaranteed by run()).
    # widths: compile-time tuple of column widths, sum(widths) == C.
    n = x.shape[0]
    out = nl.ndarray((n,), dtype=x.dtype, buffer=nl.shared_hbm)
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    x3 = x.reshape((n_prog, P, C))
    o3 = out.reshape((n_prog, P, C))
    off = 0
    for i in range(len(widths)):
        w = widths[i]
        dv = ((w * 3) // 4) // 32 * 32
        t = nl.ndarray((P, w), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=t, src=x3[pid, :, off:off + w])
        # Vector engine: first ~3/4 of the columns
        nisa.tensor_scalar(dst=t[:, 0:dv], data=t[:, 0:dv], op0=nl.multiply,
                           operand0=2.0, engine=nisa.engine.vector)
        # Scalar engine: remaining columns (exact: x*2 in fp32, cast back to fp16)
        if dv < w:
            nisa.activation(dst=t[:, dv:w], op=nl.copy, data=t[:, dv:w], scale=2.0)
        nisa.dma_copy(dst=o3[pid, :, off:off + w], src=t)
        off = off + w
    return out


def _plan(C):
    # main tile width: ~C/4, multiple of MINW, clamped to [2*MINW, MAXW]
    F = (C // 4) // MINW * MINW
    F = max(2 * MINW, min(F, MAXW))
    if F > C:
        F = C
    n_full = C // F
    left = C - n_full * F
    widths = [F] * (n_full - 1)
    tail = F + left
    # taper the final region so the exposed tail compute/store is small
    if tail >= 4 * MINW:
        a = (tail // 2) // MINW * MINW
        r = tail - a
        b = (r // 2) // MINW * MINW
        c = r - b
        widths += [a, b, c]
    elif tail >= 2 * MINW:
        a = (tail // 2) // MINW * MINW
        widths += [a, tail - a]
    else:
        widths.append(tail)
    return tuple(widths)


def run(x: torch.Tensor) -> torch.Tensor:
    global _LAST_CONFIG
    n = x.numel()
    if n % (2 * P) != 0:
        raise ValueError("mul2: n must be a multiple of 256 for this implementation")
    C = n // (2 * P)
    widths = _plan(C)
    _LAST_CONFIG = {
        "programs": 2,
        "partitions": P,
        "cols_per_partition": C,
        "tile_widths": list(widths),
        "num_tiles": len(widths),
        "vector_frac": 0.75,
        "scalar_frac": 0.25,
        "op": "multiply_2",
    }
    return mul2_kernel(x, C, widths)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
