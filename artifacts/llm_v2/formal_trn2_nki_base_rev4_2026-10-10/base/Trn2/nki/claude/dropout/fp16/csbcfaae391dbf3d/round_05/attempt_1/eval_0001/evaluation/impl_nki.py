import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_NPROG = 2
_MAIN = 4096
_PREFETCH = 3
_LAST_CONFIG = {}


# --------------------------------------------------------------------------------------
# Single committed kernel.
# Per program (physical core), its half of the data is viewed as [128, M]
# (partition-major rows, each partition owns a contiguous run of M elements) and is
# processed in column tiles of compile-time widths. Loads are issued three tiles
# ahead of compute/store (software pipelining) so DMA issue never stalls behind a
# store that is waiting on compute. Compute: (x * scale) * keep evaluated in fp32
# inside one VectorE instruction, single rounding to x.dtype at the write.
# --------------------------------------------------------------------------------------
@nki.jit
def _dropout_kernel(x, keep, scale, widths, offsets, m_expected):
    n = x.shape[0]
    P = 128
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    M = n // (n_prog * P)
    assert M == m_expected, "unexpected program count"
    xv = x.reshape((n_prog, P, M))
    kv = keep.reshape((n_prog, P, M))
    ov = out.reshape((n_prog, P, M))
    T = len(widths)

    # ---- prologue: tiles 0, 1, 2 ----
    wa = widths[0]
    oa = offsets[0]
    x0 = nl.ndarray((P, wa), dtype=x.dtype, buffer=nl.sbuf)
    k0 = nl.ndarray((P, wa), dtype=keep.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x0, src=xv[pid, :, oa:oa + wa])
    nisa.dma_copy(dst=k0, src=kv[pid, :, oa:oa + wa])
    x1 = x0
    k1 = k0
    if T > 1:
        wb = widths[1]
        ob = offsets[1]
        x1 = nl.ndarray((P, wb), dtype=x.dtype, buffer=nl.sbuf)
        k1 = nl.ndarray((P, wb), dtype=keep.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x1, src=xv[pid, :, ob:ob + wb])
        nisa.dma_copy(dst=k1, src=kv[pid, :, ob:ob + wb])
    x2 = x1
    k2 = k1
    if T > 2:
        wc = widths[2]
        oc = offsets[2]
        x2 = nl.ndarray((P, wc), dtype=x.dtype, buffer=nl.sbuf)
        k2 = nl.ndarray((P, wc), dtype=keep.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x2, src=xv[pid, :, oc:oc + wc])
        nisa.dma_copy(dst=k2, src=kv[pid, :, oc:oc + wc])

    # ---- steady state: prefetch tile j+3, compute + store tile j ----
    for j in range(T):
        x3 = x2
        k3 = k2
        if j + 3 < T:
            wd = widths[j + 3]
            od = offsets[j + 3]
            x3 = nl.ndarray((P, wd), dtype=x.dtype, buffer=nl.sbuf)
            k3 = nl.ndarray((P, wd), dtype=keep.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x3, src=xv[pid, :, od:od + wd])
            nisa.dma_copy(dst=k3, src=kv[pid, :, od:od + wd])
        wj = widths[j]
        oj = offsets[j]
        # (x * scale) * keep in fp32; dropped positions give exactly 0
        nisa.scalar_tensor_tensor(dst=x0, data=x0, op0=nl.multiply, operand0=scale,
                                  op1=nl.multiply, operand1=k0)
        nisa.dma_copy(dst=ov[pid, :, oj:oj + wj], src=x0)
        x0 = x1
        k0 = k1
        x1 = x2
        k1 = k2
        x2 = x3
        k2 = k3
    return out


# --------------------------------------------------------------------------------------
# Host-side deterministic configuration (pure function of n)
# --------------------------------------------------------------------------------------
def _plan_for(n):
    """Column-tile widths per program over M = n / (2*128) columns per partition."""
    M = n // (_NPROG * _P)
    if M % _MAIN == 0 and M >= 2 * _MAIN:
        head = [512, 1024, 2048]                 # 3584 columns
        tail = [2560, 1024, 512, 256, 256]       # 4608 columns
        nmid = (M - 2 * _MAIN) // _MAIN
        widths = head + [_MAIN] * nmid + tail
        cmain = _MAIN
    elif M == _MAIN:
        widths = [512, 1024, 1024, 768, 512, 256]
        cmain = 1024
    else:
        cu = 2048
        while cu > 1 and (M % cu) != 0:
            cu //= 2
        widths = [cu] * (M // cu)
        cmain = cu
    offsets = []
    off = 0
    for w in widths:
        offsets.append(off)
        off += w
    assert off == M
    return tuple(int(w) for w in widths), tuple(int(o) for o in offsets), int(M), int(cmain)


def run(x, x_keep, p):
    global _LAST_CONFIG
    n = int(x.numel())
    assert n % (_NPROG * _P) == 0, "n must be a multiple of 256"
    scale = 1.0 / (1.0 - float(p))
    widths, offsets, M, cmain = _plan_for(n)

    out = _dropout_kernel(x, x_keep, scale, widths, offsets, M)

    _LAST_CONFIG = {
        "kernel": "stt_pipelined",
        "programs": _NPROG,
        "partitions": _P,
        "layout": "partition_major_per_program",
        "cols_per_partition": M,
        "main_tile": cmain,
        "num_tiles": len(widths),
        "tile_widths": list(widths),
        "prefetch_depth": _PREFETCH,
        "scale": float(scale),
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
