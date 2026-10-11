import nki
import nki.language as nl
import nki.isa as nisa

_P = 128
_LAST_CONFIG = {}


@nki.jit
def _swiglu_kernel(x, y, tile_f):
    # out = silu(x) * y, fused single pass; fp32 math, one cast at the end.
    M, N = x.shape
    P = 128
    out = nl.ndarray((M, N), dtype=x.dtype, buffer=nl.shared_hbm)

    W = (M * N) // P                      # elements per partition (whole tensor)
    n_prog = nl.num_programs()
    pid = nl.program_id(0)

    nsplit = 1
    part = 0
    if W % n_prog == 0:
        nsplit = n_prog
        part = pid
    Wp = W // nsplit                      # elements per partition for this program

    xv = x.reshape((nsplit, P, Wp))
    yv = y.reshape((nsplit, P, Wp))
    ov = out.reshape((nsplit, P, Wp))

    ntiles = (Wp + tile_f - 1) // tile_f
    for i in range(ntiles):
        c0 = i * tile_f
        f = min(tile_f, Wp - c0)

        xt = nl.ndarray((P, f), dtype=x.dtype, buffer=nl.sbuf)
        yt = nl.ndarray((P, f), dtype=y.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=xt, src=xv[part, :, c0:c0 + f])
        nisa.dma_copy(dst=yt, src=yv[part, :, c0:c0 + f])

        # silu(x) evaluated in fp32 on ScalarE, kept in an fp32 tile
        st = nl.ndarray((P, f), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=st, op=nl.silu, data=xt)

        # fp32 multiply by y on VectorE, single downcast to the output dtype
        # (result written into the now-dead x tile to save SBUF)
        nisa.tensor_tensor(dst=xt, data1=st, data2=yt, op=nl.multiply)

        nisa.dma_copy(dst=ov[part, :, c0:c0 + f], src=xt)
    return out


def _choose_tile(wp):
    # deterministic: aim for >= ~32 tiles per program, tile width in [1024, 4096]
    target = wp // 32
    f = 1024
    while f < target and f < 4096:
        f *= 2
    if wp > 0 and f > wp:
        f = wp
    if f < 1:
        f = 1
    return f


def run(x, y):
    global _LAST_CONFIG
    assert x.shape == y.shape, "x and y must have the same shape"
    assert x.dim() == 2, "expected 2-D inputs (M, N)"
    M, N = int(x.shape[0]), int(x.shape[1])
    E = M * N
    assert E % _P == 0, "M*N must be a multiple of 128"

    n_prog = 2  # bare kernel call runs with LNC=2 on this host
    W = E // _P
    if W % n_prog == 0:
        wp = W // n_prog
    else:
        wp = W
    tile_f = _choose_tile(wp)

    out = _swiglu_kernel(x, y, tile_f)

    _LAST_CONFIG = {
        "layout": "flat_view(programs,128,W)",
        "partitions": _P,
        "programs": n_prog,
        "elems_per_partition_per_program": wp,
        "tile_free": tile_f,
        "num_tiles_per_program": (wp + tile_f - 1) // tile_f,
        "silu_engine": "scalar(activation silu, fp32 out)",
        "mul_engine": "vector(tensor_tensor, fp32 math, fp16 out)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
