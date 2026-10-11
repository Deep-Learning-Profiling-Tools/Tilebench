import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_NPROG = 2
_LAST_CONFIG = {}
# API-capability selection on compile failure only (never by timing):
# 0 = software-pipelined scalar_tensor_tensor (prefetch depth 2)
# 1 = scalar_tensor_tensor, plain loop (round-2 kernel, known good)
# 2 = two-pass (fp32 intermediate tile)
# 3 = round-1 kernel
_MODE = 0
_MODE_NAMES = ("stt_pipelined", "stt_default", "two_pass", "legacy_round1")


# --------------------------------------------------------------------------------------
# Mode 0: per program, data viewed as [128, M] (partition-major rows), processed in
# column tiles of compile-time widths. Loads run two tiles ahead of compute/store in
# program order so DMA issue never stalls behind a store waiting on compute.
# --------------------------------------------------------------------------------------
@nki.jit
def _k_pipe(x, keep, scale, widths, offsets, m_expected):
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

    w0 = widths[0]
    c0 = offsets[0]
    xa = nl.ndarray((P, w0), dtype=x.dtype, buffer=nl.sbuf)
    ka = nl.ndarray((P, w0), dtype=keep.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=xa, src=xv[pid, :, c0:c0 + w0])
    nisa.dma_copy(dst=ka, src=kv[pid, :, c0:c0 + w0])
    xb = xa
    kb = ka
    if T > 1:
        w1 = widths[1]
        c1 = offsets[1]
        xb = nl.ndarray((P, w1), dtype=x.dtype, buffer=nl.sbuf)
        kb = nl.ndarray((P, w1), dtype=keep.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xb, src=xv[pid, :, c1:c1 + w1])
        nisa.dma_copy(dst=kb, src=kv[pid, :, c1:c1 + w1])
    for j in range(T):
        xc = xb
        kc = kb
        if j + 2 < T:
            w2 = widths[j + 2]
            c2 = offsets[j + 2]
            xc = nl.ndarray((P, w2), dtype=x.dtype, buffer=nl.sbuf)
            kc = nl.ndarray((P, w2), dtype=keep.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xc, src=xv[pid, :, c2:c2 + w2])
            nisa.dma_copy(dst=kc, src=kv[pid, :, c2:c2 + w2])
        wj = widths[j]
        cj = offsets[j]
        # (x * scale) * keep evaluated in fp32, single rounding to x.dtype
        nisa.scalar_tensor_tensor(dst=xa, data=xa, op0=nl.multiply, operand0=scale,
                                  op1=nl.multiply, operand1=ka)
        nisa.dma_copy(dst=ov[pid, :, cj:cj + wj], src=xa)
        xa = xb
        ka = kb
        xb = xc
        kb = kc
    return out


# --------------------------------------------------------------------------------------
# Mode 1: round-2 kernel (known good)
# --------------------------------------------------------------------------------------
@nki.jit
def _k_stt(x, keep, scale, widths, offsets, m_expected):
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
    for j in range(len(widths)):
        c0 = offsets[j]
        cj = widths[j]
        xt = nl.ndarray((P, cj), dtype=x.dtype, buffer=nl.sbuf)
        kt = nl.ndarray((P, cj), dtype=keep.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=xv[pid, :, c0:c0 + cj])
        nisa.dma_copy(dst=kt, src=kv[pid, :, c0:c0 + cj])
        nisa.scalar_tensor_tensor(dst=xt, data=xt, op0=nl.multiply, operand0=scale,
                                  op1=nl.multiply, operand1=kt)
        nisa.dma_copy(dst=ov[pid, :, c0:c0 + cj], src=xt)
    return out


# --------------------------------------------------------------------------------------
# Mode 2: two-pass (fp32 intermediate tile on chip)
# --------------------------------------------------------------------------------------
@nki.jit
def _k_two_pass(x, keep, scale, widths, offsets, m_expected):
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
    for j in range(len(widths)):
        c0 = offsets[j]
        cj = widths[j]
        xt = nl.ndarray((P, cj), dtype=x.dtype, buffer=nl.sbuf)
        kt = nl.ndarray((P, cj), dtype=keep.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=xv[pid, :, c0:c0 + cj])
        nisa.dma_copy(dst=kt, src=kv[pid, :, c0:c0 + cj])
        tmp = nl.ndarray((P, cj), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tmp, data1=xt, data2=kt, op=nl.multiply,
                           engine=nisa.engine.vector)
        nisa.activation(dst=xt, op=nl.copy, data=tmp, scale=scale)
        nisa.dma_copy(dst=ov[pid, :, c0:c0 + cj], src=xt)
    return out


# --------------------------------------------------------------------------------------
# Mode 3: round-1 kernel (last fallback)
# --------------------------------------------------------------------------------------
@nki.jit
def _k_legacy(x, keep, scale, C):
    n = x.shape[0]
    P = 128
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    T = n // (P * C)
    x3 = x.reshape((T, P, C))
    k3 = keep.reshape((T, P, C))
    o3 = out.reshape((T, P, C))
    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    Tp = T // n_prog
    for i in range(Tp):
        t = pid * Tp + i
        xt = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        kt = nl.ndarray((P, C), dtype=keep.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x3[t])
        nisa.dma_copy(dst=kt, src=k3[t])
        tmp = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tmp, data1=xt, data2=kt, op=nl.multiply,
                           engine=nisa.engine.vector)
        nisa.activation(dst=xt, op=nl.copy, data=tmp, scale=scale)
        nisa.dma_copy(dst=o3[t], src=xt)
    return out


# --------------------------------------------------------------------------------------
# Host-side deterministic configuration
# --------------------------------------------------------------------------------------
def _plan_for(n):
    """Column-tile widths per program over M = n / (2*128) columns per partition."""
    M = n // (_NPROG * _P)
    cmain = 4096
    if M % cmain == 0 and M >= 2 * cmain:
        head = [512, 1024, 2048]
        tail = [3072, 1024, 512]
        nmid = (M - 2 * cmain) // cmain
        widths = head + [cmain] * nmid + tail
    elif M == cmain:
        widths = [512, 1024, 1024, 1024, 512]
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


def _legacy_c(n):
    m = n // (2 * _P)
    c = min(2048, max(256, m // 4))
    while c > 1 and (m % c) != 0:
        c //= 2
    return int(c)


def _call(mode, x, x_keep, scale, widths, offsets, M, n):
    if mode == 0:
        return _k_pipe(x, x_keep, scale, widths, offsets, M)
    if mode == 1:
        return _k_stt(x, x_keep, scale, widths, offsets, M)
    if mode == 2:
        return _k_two_pass(x, x_keep, scale, widths, offsets, M)
    return _k_legacy(x, x_keep, scale, _legacy_c(n))


def run(x, x_keep, p):
    global _LAST_CONFIG, _MODE
    n = int(x.numel())
    assert n % (_NPROG * _P) == 0, "n must be a multiple of 256"
    scale = 1.0 / (1.0 - float(p))
    widths, offsets, M, cmain = _plan_for(n)

    out = None
    mode = _MODE
    while True:
        mode = _MODE
        if mode >= 3:
            out = _call(3, x, x_keep, scale, widths, offsets, M, n)
            break
        try:
            out = _call(mode, x, x_keep, scale, widths, offsets, M, n)
            break
        except Exception:
            _MODE = mode + 1

    if mode >= 3:
        C = _legacy_c(n)
        cfg = {
            "mode": _MODE_NAMES[3],
            "programs": _NPROG,
            "partitions": _P,
            "tile_free": C,
            "tiles_per_program": int((n // (2 * _P)) // C),
            "scale": float(scale),
        }
    else:
        cfg = {
            "mode": _MODE_NAMES[mode],
            "programs": _NPROG,
            "partitions": _P,
            "layout": "partition_major_per_program",
            "cols_per_partition": M,
            "main_tile": cmain,
            "num_tiles": len(widths),
            "tile_widths": list(widths),
            "prefetch_depth": 2 if mode == 0 else 0,
            "scale": float(scale),
        }
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
