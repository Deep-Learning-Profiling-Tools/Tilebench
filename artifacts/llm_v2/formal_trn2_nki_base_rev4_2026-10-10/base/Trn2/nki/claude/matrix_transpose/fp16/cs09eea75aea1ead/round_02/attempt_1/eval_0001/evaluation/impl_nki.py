import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Partition tile size (TensorE transpose block is P x P).
P = 128
# Fixed configuration literals.
_IN_SLOTS = 3        # SBUF input chunk slots -> prefetch depth 2 chunks
_OUT_SLOTS = 8       # SBUF output staging slots
_W_CAP = 2048        # output tile width = number of x rows per transposed tile
_BC_CAP = 1024       # input chunk width = number of x columns per chunk
_G_CAP = 4           # row tiles (of 128 rows) per load DMA
_PSUM_GROUP = 2      # PSUM tiles allocated together (each 2 banks)

_LAST_CONFIG = {}


def _pick_div(d, cap):
    c = cap
    while c >= P:
        if d % c == 0:
            return c
        c = c // 2
    return 0


def _pick_group(rt, cap):
    g = cap
    while g > 1:
        if rt % g == 0:
            return g
        g = g // 2
    return 1


# ----------------------------------------------------------------------------
# kernel-side helpers (module level, used inside the @nki.jit kernel)
# ----------------------------------------------------------------------------
def _sel3(s, a, b, c):
    r = c
    if s == 0:
        r = a
    if s == 1:
        r = b
    return r


def _sel8(s, a0, a1, a2, a3, a4, a5, a6, a7):
    r = a7
    if s == 0:
        r = a0
    if s == 1:
        r = a1
    if s == 2:
        r = a2
    if s == 3:
        r = a3
    if s == 4:
        r = a4
    if s == 5:
        r = a5
    if s == 6:
        r = a6
    return r


def _load_unit(x, dst, rb, c0, n, bc, rt, G):
    # dst[p, k, q] = x[rb + k*P + p, c0 + q]   (one DMA per G row tiles)
    for gi in range(rt // G):
        k0 = gi * G
        src = x.ap(pattern=[[n, P], [P * n, G], [1, bc]],
                   offset=(rb + k0 * P) * n + c0)
        nisa.dma_copy(dst=dst[:, k0:k0 + G, :], src=src)


def _transpose_into(ps, xin2, jc, rt, bc):
    # ps[q, k*P + p] = xin2[p, k*bc + jc*P + q]
    for k in range(rt):
        nisa.nc_transpose(dst=ps[:, k * P:(k + 1) * P],
                          data=xin2[:, k * bc + jc * P:k * bc + (jc + 1) * P],
                          engine=nisa.engine.tensor)


def _evict_store(out, ps, ot, orow, ocol, W):
    h = W // 2
    nisa.tensor_copy(dst=ot[:, 0:h], src=ps[:, 0:h], engine=nisa.engine.vector)
    nisa.tensor_copy(dst=ot[:, h:W], src=ps[:, h:W], engine=nisa.engine.scalar)
    nisa.dma_copy(dst=out[orow:orow + P, ocol:ocol + W], src=ot)


@nki.jit
def _transpose_kernel(x, W, bc, G):
    m = x.shape[0]
    n = x.shape[1]
    dt = x.dtype
    out = nl.ndarray((n, m), dtype=dt, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R = m // n_prog            # x rows handled by this program
    row0 = pid * R
    nsb = R // W               # row sub-blocks per program
    nch = n // bc              # column chunks
    nu = nsb * nch             # work units per program
    rt = W // P                # row tiles per unit
    ct = bc // P               # column tiles per unit

    # explicit SBUF input slots (prefetch ring)
    in0 = nl.ndarray((P, rt, bc), dtype=dt, buffer=nl.sbuf)
    in1 = nl.ndarray((P, rt, bc), dtype=dt, buffer=nl.sbuf)
    in2 = nl.ndarray((P, rt, bc), dtype=dt, buffer=nl.sbuf)

    # explicit SBUF output staging slots
    o0 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o1 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o2 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o3 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o4 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o5 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o6 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o7 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)

    # prologue: prefetch the first two units
    if nu > 0:
        _load_unit(x, in0, row0 + (0 % nsb) * W, (0 // nsb) * bc, n, bc, rt, G)
    if nu > 1:
        _load_unit(x, in1, row0 + (1 % nsb) * W, (1 // nsb) * bc, n, bc, rt, G)

    for u in range(nu):
        if u + 2 < nu:
            v = u + 2
            _load_unit(x, _sel3(v % 3, in0, in1, in2),
                       row0 + (v % nsb) * W, (v // nsb) * bc, n, bc, rt, G)
        xin = _sel3(u % 3, in0, in1, in2)
        xin2 = xin.reshape((P, rt * bc))
        rb = row0 + (u % nsb) * W
        c0 = (u // nsb) * bc
        if ct % 2 == 0:
            for jp in range(ct // 2):
                ja = 2 * jp
                jb = ja + 1
                ga = u * ct + ja
                pa = nl.ndarray((P, W), dtype=dt, buffer=nl.psum)
                pb = nl.ndarray((P, W), dtype=dt, buffer=nl.psum)
                _transpose_into(pa, xin2, ja, rt, bc)
                _transpose_into(pb, xin2, jb, rt, bc)
                _evict_store(out, pa,
                             _sel8(ga % 8, o0, o1, o2, o3, o4, o5, o6, o7),
                             c0 + ja * P, rb, W)
                _evict_store(out, pb,
                             _sel8((ga + 1) % 8, o0, o1, o2, o3, o4, o5, o6, o7),
                             c0 + jb * P, rb, W)
        else:
            for jc in range(ct):
                g = u * ct + jc
                ps = nl.ndarray((P, W), dtype=dt, buffer=nl.psum)
                _transpose_into(ps, xin2, jc, rt, bc)
                _evict_store(out, ps,
                             _sel8(g % 8, o0, o1, o2, o3, o4, o5, o6, o7),
                             c0 + jc * P, rb, W)
    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    m = int(x.shape[0])
    n = int(x.shape[1])
    if m % (2 * P) != 0 or n % P != 0:
        raise ValueError("matrix_transpose: unsupported shape for this kernel")
    W = _pick_div(m // 2, _W_CAP)
    bc = _pick_div(n, _BC_CAP)
    if W == 0 or bc == 0:
        raise ValueError("matrix_transpose: unsupported shape for this kernel")
    G = _pick_group(W // P, _G_CAP)
    out = _transpose_kernel(x, W, bc, G)
    _LAST_CONFIG = {
        "out_tile_width": W,
        "chunk_cols": bc,
        "rows_per_load_dma": G * P,
        "transpose_tile": P,
        "transpose_engine": "tensor (nc_transpose, fp16 psum)",
        "in_slots": _IN_SLOTS,
        "prefetch_units": _IN_SLOTS - 1,
        "out_slots": _OUT_SLOTS,
        "psum_group": _PSUM_GROUP if (bc // P) % 2 == 0 else 1,
        "evict_engines": "vector+scalar split per tile (one bank each)",
        "core_split": "x row halves per program (LNC=2)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
