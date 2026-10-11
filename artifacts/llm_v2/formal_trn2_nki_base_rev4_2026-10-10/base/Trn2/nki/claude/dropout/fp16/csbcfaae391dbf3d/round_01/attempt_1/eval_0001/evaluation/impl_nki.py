import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_LAST_CONFIG = {}


@nki.jit
def _dropout_kernel(x, keep, scale, C):
    # out = (x * keep) * scale, computed in fp32, rounded once to x.dtype at the end.
    n = x.shape[0]
    P = 128
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)

    T = n // (P * C)                 # total number of [P, C] tiles (contiguous blocks)
    x3 = x.reshape((T, P, C))
    k3 = keep.reshape((T, P, C))
    o3 = out.reshape((T, P, C))

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    Tp = T // n_prog                 # tiles per program (T is even in the task domain)

    for i in range(Tp):
        t = pid * Tp + i
        xt = nl.ndarray((P, C), dtype=x.dtype, buffer=nl.sbuf)
        kt = nl.ndarray((P, C), dtype=keep.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=x3[t])
        nisa.dma_copy(dst=kt, src=k3[t])

        # mask application in fp32 (exact: mask is 0/1), fp32 intermediate
        tmp = nl.ndarray((P, C), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tmp, data1=xt, data2=kt, op=nl.multiply,
                           engine=nisa.engine.vector)

        # rescale in fp32 on ScalarE, single rounding to x.dtype (reuse xt as output)
        nisa.activation(dst=xt, op=nl.copy, data=tmp, scale=scale)

        nisa.dma_copy(dst=o3[t], src=xt)
    return out


def _config_for(n):
    # elements per partition per program (2 programs under LNC=2)
    m = n // (2 * _P)
    c = min(2048, max(256, m // 4))
    while c > 1 and (m % c) != 0:
        c //= 2
    return {
        "tile_free": int(c),
        "partitions": _P,
        "programs": 2,
        "tiles_per_program": int(m // c) if c > 0 else 0,
        "intermediate_dtype": "float32",
        "engines": "vector(mask)+scalar(rescale)",
    }


def run(x, x_keep, p):
    global _LAST_CONFIG
    n = int(x.numel())
    assert n % (2 * _P) == 0, "n must be a multiple of 256"
    cfg = _config_for(n)
    C = cfg["tile_free"]
    assert n % (2 * _P * C) == 0, "unsupported n for chosen tiling"
    scale = 1.0 / (1.0 - float(p))
    out = _dropout_kernel(x, x_keep, scale, C)
    _LAST_CONFIG = dict(cfg)
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
