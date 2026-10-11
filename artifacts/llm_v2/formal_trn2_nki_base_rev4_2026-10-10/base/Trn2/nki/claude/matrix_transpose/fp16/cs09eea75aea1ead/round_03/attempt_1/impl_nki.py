import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Partition tile size (TensorE transpose block is P x P).
P = 128

# Fixed configuration literals.
_W_CAP = 2048          # output tile width = x rows per output tile (per program)
_BC_CAP = 1024         # x columns per chunk (8 column tiles)
_G_CAP = 4             # row tiles per load DMA / per progressive transpose group
_IN_SLOTS = 6          # SBUF input group slots (ring)
_PREFETCH = 4          # load prefetch distance in groups (slot reuse slack = 2 groups)
_OUT_UNIT_SLOTS = 3    # SBUF output staging slots, each = one full unit (8 tiles)

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


def _sel6(s, a0, a1, a2, a3, a4, a5):
    r = a5
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


def _load_group(x, dst, t, row0, nsb, ngr, W, bc, G, n):
    # group t -> unit u (row sub-block u % nsb, column chunk u // nsb), group g in unit
    # dst[p, k, q] = x[r0 + k*P + p, c0 + q]
    u = t // ngr
    g = t % ngr
    rb = row0 + (u % nsb) * W
    c0 = (u // nsb) * bc
    r0 = rb + g * G * P
    src = x.ap(pattern=[[n, P], [P * n, G], [1, bc]], offset=r0 * n + c0)
    nisa.dma_copy(dst=dst, src=src)


@nki.jit
def _transpose_kernel(x, W, bc, G, NI, PD, NS):
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
    nu = nsb * nch             # units per program
    rt = W // P                # row tiles per unit
    ngr = rt // G              # load groups per unit
    ct = bc // P               # column tiles per unit
    GW = G * P                 # output columns produced per group
    NT = nu * ngr              # total groups per program

    # SBUF input group slots
    i0 = nl.ndarray((P, G, bc), dtype=dt, buffer=nl.sbuf)
    i1 = nl.ndarray((P, G, bc), dtype=dt, buffer=nl.sbuf)
    i2 = nl.ndarray((P, G, bc), dtype=dt, buffer=nl.sbuf)
    i3 = nl.ndarray((P, G, bc), dtype=dt, buffer=nl.sbuf)
    i4 = nl.ndarray((P, G, bc), dtype=dt, buffer=nl.sbuf)
    i5 = nl.ndarray((P, G, bc), dtype=dt, buffer=nl.sbuf)

    # SBUF output staging: 3 unit slots x 8 column tiles, separate tensors
    o00 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o01 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o02 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o03 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o04 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o05 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o06 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o07 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o10 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o11 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o12 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o13 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o14 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o15 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o16 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o17 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o20 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o21 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o22 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o23 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o24 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o25 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o26 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o27 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)

    # prologue: prefetch the first PD groups
    npre = min(PD, NT)
    for t in range(npre):
        _load_group(x, _sel6(t % NI, i0, i1, i2, i3, i4, i5),
                    t, row0, nsb, ngr, W, bc, G, n)

    for t in range(NT):
        if t + PD < NT:
            tn = t + PD
            _load_group(x, _sel6(tn % NI, i0, i1, i2, i3, i4, i5),
                        tn, row0, nsb, ngr, W, bc, G, n)
        u = t // ngr
        g = t % ngr
        rb = row0 + (u % nsb) * W
        c0 = (u // nsb) * bc
        xin = _sel6(t % NI, i0, i1, i2, i3, i4, i5)
        xin2 = xin.reshape((P, G * bc))
        osl = u % NS
        for j in range(ct):
            ps = nl.ndarray((P, GW), dtype=dt, buffer=nl.psum)
            for k in range(G):
                # ps[q, k*P + p] = x[r0 + k*P + p, c0 + j*P + q]
                nisa.nc_transpose(dst=ps[:, k * P:(k + 1) * P],
                                  data=xin2[:, k * bc + j * P:k * bc + (j + 1) * P],
                                  engine=nisa.engine.tensor)
            ot = _sel3(osl,
                       _sel8(j, o00, o01, o02, o03, o04, o05, o06, o07),
                       _sel8(j, o10, o11, o12, o13, o14, o15, o16, o17),
                       _sel8(j, o20, o21, o22, o23, o24, o25, o26, o27))
            if (t * ct + j) % 2 == 0:
                nisa.tensor_copy(dst=ot[:, g * GW:(g + 1) * GW], src=ps,
                                 engine=nisa.engine.vector)
            else:
                nisa.tensor_copy(dst=ot[:, g * GW:(g + 1) * GW], src=ps,
                                 engine=nisa.engine.scalar)
            if g == ngr - 1:
                # out[c0 + j*P + q, rb + w] = ot[q, w]
                nisa.dma_copy(dst=out[c0 + j * P:c0 + (j + 1) * P, rb:rb + W],
                              src=ot)
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
    out = _transpose_kernel(x, W, bc, G, _IN_SLOTS, _PREFETCH, _OUT_UNIT_SLOTS)
    _LAST_CONFIG = {
        "out_tile_width": W,
        "chunk_cols": bc,
        "rows_per_group": G * P,
        "transpose_tile": P,
        "transpose_engine": "tensor (nc_transpose, fp16 psum)",
        "psum_tile_cols": G * P,
        "in_slots": _IN_SLOTS,
        "prefetch_groups": _PREFETCH,
        "out_unit_slots": _OUT_UNIT_SLOTS,
        "evict_engines": "vector/scalar alternating per column tile",
        "pipeline": "progressive per-group transpose+evict, store on last group",
        "core_split": "x row halves per program (LNC=2)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
