import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_NPROG = 2
_LAST_CONFIG = {}
# API-capability selection (decided once, never by timing):
# 0 = scalar_tensor_tensor on VectorE, 1 = scalar_tensor_tensor (compiler-chosen engine),
# 2 = two-pass (fp32 intermediate), 3 = round-1 kernel.
_MODE = 0
_MODE_NAMES = ("stt_vector", "stt_default", "two_pass", "legacy_round1")


# --------------------------------------------------------------------------------------
# Main kernels: per program, data viewed as [128, M] (partition-major, contiguous rows),
# processed in column tiles of compile-time widths (ramp-up / bulk / ramp-down).
# --------------------------------------------------------------------------------------
@nki.jit
def _k_stt_vec(x, keep, scale, widths, offsets, m_expected):
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
        # (x * scale) * keep, evaluated in fp32, single rounding to x.dtype
        nisa.scalar_tensor_tensor(dst=xt, data=xt, op0=nl.multiply, operand0=scale,
                                  op1=nl.multiply, operand1=kt,
                                  engine=nisa.engine.vector)
        nisa.dma_copy(dst=ov[pid, :, c0:c0 + cj], src=xt)
    return out


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
# Round-1 kernel (known-good last fallback)
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
    while cmain > 1024 and M < 4 * cmain:
        cmain //= 2
    head = []
    c = 512
    while c < cmain:
        head.append(c)
        c *= 2
    hsum = sum(head)
    rem = M - 2 * hsum
    widths = []
    if M % 512 == 0 and rem >= cmain:
        nmain = rem // cmain
        left = rem - nmain * cmain          # multiple of 512, < cmain
        widths = list(head) + [cmain] * nmain + list(reversed(head)) + [512] * (left // 512)
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
        return _k_stt_vec(x, x_keep, scale, widths, offsets, M)
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
            "scale": float(scale),
        }
    _LAST_CONFIG = cfg
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
