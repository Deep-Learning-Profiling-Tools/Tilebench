import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Partition tile size (TensorE transpose block is P x P).
P = 128

# Fixed configuration literals.
_W_CAP = 2048          # output tile width = x rows per output tile (per program)
_BC = 2048             # x columns per chunk (16 column tiles); last chunk may be narrower
_G_CAP = 4             # row tiles per progressive transpose group
_LOAD_SPLIT = 4        # DMAs per group load (one per row tile)
_PREFETCH = 3          # load prefetch distance in groups
_IN_SLOTS = 4          # SBUF input group slots (ring) -- fixed by the kernel code
_OUT_UNIT_SLOTS = 2    # SBUF output staging unit slots -- fixed by the kernel code

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
def _sel2(s, a, b):
    r = b
    if s == 0:
        r = a
    return r


def _sel4(s, a, b, c, d):
    r = d
    if s == 0:
        r = a
    if s == 1:
        r = b
    if s == 2:
        r = c
    return r


def _sel16(s, a0, a1, a2, a3, a4, a5, a6, a7,
           a8, a9, a10, a11, a12, a13, a14, a15):
    r = a15
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
    if s == 7:
        r = a7
    if s == 8:
        r = a8
    if s == 9:
        r = a9
    if s == 10:
        r = a10
    if s == 11:
        r = a11
    if s == 12:
        r = a12
    if s == 13:
        r = a13
    if s == 14:
        r = a14
    return r


def _load_group(x, slot, t, row0, nsb, ngr, W, BC, G, NH, n, nch, shift):
    # group t -> unit u (row sub-block u % nsb, rotated column chunk), group g in unit
    # slot[p, k, q] = x[r0 + k*P + p, c0 + q]   for q < w
    u = t // ngr
    g = t % ngr
    rb = row0 + (u % nsb) * W
    ci = (u // nsb + shift) % nch
    c0 = ci * BC
    w = min(BC, n - c0)
    r0 = rb + g * G * P
    kh = G // NH
    for h in range(NH):
        src = x.ap(pattern=[[n, P], [P * n, kh], [1, w]],
                   offset=(r0 + h * kh * P) * n + c0)
        nisa.dma_copy(dst=slot[:, h * kh:(h + 1) * kh, 0:w], src=src)


@nki.jit
def _transpose_kernel(x, W, BC, G, NH, PD):
    m = x.shape[0]
    n = x.shape[1]
    dt = x.dtype
    out = nl.ndarray((n, m), dtype=dt, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    R = m // n_prog            # x rows handled by this program
    row0 = pid * R
    nsb = R // W               # row sub-blocks per program
    nch = (n + BC - 1) // BC   # column chunks
    shift = pid % nch          # stagger chunk order between programs
    nu = nsb * nch             # units per program
    rt = W // P                # row tiles per unit
    ngr = rt // G              # groups per unit
    GW = G * P                 # output columns produced per group
    NT = nu * ngr              # total groups per program

    # SBUF input group slots (4)
    i0 = nl.ndarray((P, G, BC), dtype=dt, buffer=nl.sbuf)
    i1 = nl.ndarray((P, G, BC), dtype=dt, buffer=nl.sbuf)
    i2 = nl.ndarray((P, G, BC), dtype=dt, buffer=nl.sbuf)
    i3 = nl.ndarray((P, G, BC), dtype=dt, buffer=nl.sbuf)

    # SBUF output staging: 2 unit slots x 16 column tiles, separate tensors
    o00 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o01 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o02 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o03 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o04 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o05 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o06 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o07 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o08 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o09 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o0a = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o0b = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o0c = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o0d = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o0e = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o0f = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o10 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o11 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o12 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o13 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o14 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o15 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o16 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o17 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o18 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o19 = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o1a = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o1b = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o1c = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o1d = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o1e = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)
    o1f = nl.ndarray((P, W), dtype=dt, buffer=nl.sbuf)

    # prologue: prefetch the first PD groups
    npre = min(PD, NT)
    for t in range(npre):
        _load_group(x, _sel4(t % 4, i0, i1, i2, i3),
                    t, row0, nsb, ngr, W, BC, G, NH, n, nch, shift)

    for t in range(NT):
        if t + PD < NT:
            tn = t + PD
            _load_group(x, _sel4(tn % 4, i0, i1, i2, i3),
                        tn, row0, nsb, ngr, W, BC, G, NH, n, nch, shift)
        u = t // ngr
        g = t % ngr
        rb = row0 + (u % nsb) * W
        ci = (u // nsb + shift) % nch
        c0 = ci * BC
        w = min(BC, n - c0)
        ctw = w // P
        xin = _sel4(t % 4, i0, i1, i2, i3)
        xin2 = xin.reshape((P, G * BC))
        osl = u % 2
        for j in range(ctw):
            ps = nl.ndarray((P, GW), dtype=dt, buffer=nl.psum)
            for k in range(G):
                # ps[q, k*P + p] = x[r0 + k*P + p, c0 + j*P + q]
                nisa.nc_transpose(dst=ps[:, k * P:(k + 1) * P],
                                  data=xin2[:, k * BC + j * P:k * BC + (j + 1) * P],
                                  engine=nisa.engine.tensor)
            ot = _sel2(osl,
                       _sel16(j, o00, o01, o02, o03, o04, o05, o06, o07,
                              o08, o09, o0a, o0b, o0c, o0d, o0e, o0f),
                       _sel16(j, o10, o11, o12, o13, o14, o15, o16, o17,
                              o18, o19, o1a, o1b, o1c, o1d, o1e, o1f))
            if j % 2 == 0:
                nisa.tensor_copy(dst=ot[:, g * GW:(g + 1) * GW], src=ps,
                                 engine=nisa.engine.vector)
            else:
                nisa.tensor_copy(dst=ot[:, g * GW:(g + 1) * GW], src=ps,
                                 engine=nisa.engine.scalar)
            if g == ngr - 1:
                # out[c0 + j*P + q, rb + w'] = ot[q, w']
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
    if W == 0:
        raise ValueError("matrix_transpose: unsupported shape for this kernel")
    G = _pick_group(W // P, _G_CAP)
    NH = _LOAD_SPLIT
    while NH > 1 and G % NH != 0:
        NH = NH // 2
    BC = _BC
    out = _transpose_kernel(x, W, BC, G, NH, _PREFETCH)
    nch = (n + BC - 1) // BC
    last_w = n - (nch - 1) * BC
    _LAST_CONFIG = {
        "out_tile_width": W,
        "chunk_cols": BC,
        "last_chunk_cols": last_w,
        "rows_per_group": G * P,
        "dmas_per_group": NH,
        "transpose_tile": P,
        "transpose_engine": "tensor (nc_transpose, fp16 psum)",
        "psum_tile_cols": G * P,
        "in_slots": _IN_SLOTS,
        "prefetch_groups": _PREFETCH,
        "out_unit_slots": _OUT_UNIT_SLOTS,
        "evict_engines": "vector/scalar alternating per column tile",
        "pipeline": "progressive per-group transpose+evict, store on last group",
        "core_split": "x row halves per program (LNC=2)",
        "chunk_order": "rotated by program id (stagger)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
