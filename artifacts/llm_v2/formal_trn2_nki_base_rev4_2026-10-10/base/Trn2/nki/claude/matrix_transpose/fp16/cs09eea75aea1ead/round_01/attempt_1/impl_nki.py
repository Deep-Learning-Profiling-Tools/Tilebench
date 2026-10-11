import torch
import nki
import nki.language as nl
import nki.isa as nisa

# Partition tile size (TensorE transpose block is P x P).
P = 128
# Fixed pipeline configuration (literals).
_IN_SLOTS = 3        # SBUF input block slots -> prefetch depth 2 blocks
_PSUM_GROUP = 4      # PSUM tiles (one bank each) written before eviction
_OUT_SLOTS = 16      # SBUF output staging slots

_LAST_CONFIG = {}


def _pick_block(d):
    # deterministic block size: largest of these that divides the dimension
    for c in (1024, 512, 256, 128):
        if d % c == 0:
            return c
    raise ValueError("matrix_transpose: dimensions must be multiples of 128")


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


def _pick_out(g, o0, o1, o2, o3, o4, o5, o6, o7,
              o8, o9, o10, o11, o12, o13, o14, o15):
    s = g % 16
    r = o15
    if s < 8:
        r = _sel8(s, o0, o1, o2, o3, o4, o5, o6, o7)
    else:
        r = _sel8(s - 8, o8, o9, o10, o11, o12, o13, o14, o15)
    return r


def _load_block(x, dst, b, ni, br, bc):
    # dst[p, k*bc + q] = x[r0 + k*P + p, c0 + q]
    r0 = (b % ni) * br
    c0 = (b // ni) * bc
    for k in range(br // P):
        nisa.dma_copy(dst=dst[:, k * bc:(k + 1) * bc],
                      src=x[r0 + k * P:r0 + (k + 1) * P, c0:c0 + bc])


def _transpose_into(ps, xin, jc, rt, bc):
    # ps[q, k*P + p] = xin[p, k*bc + jc*P + q]
    for k in range(rt):
        nisa.nc_transpose(dst=ps[:, k * P:(k + 1) * P],
                          data=xin[:, k * bc + jc * P:k * bc + (jc + 1) * P],
                          engine=nisa.engine.tensor)


def _evict_store(out, ps, ot, use_vector, orow, ocol, br):
    if use_vector:
        nisa.tensor_copy(dst=ot, src=ps, engine=nisa.engine.vector)
    else:
        nisa.tensor_copy(dst=ot, src=ps, engine=nisa.engine.scalar)
    nisa.dma_copy(dst=out[orow:orow + P, ocol:ocol + br], src=ot)


@nki.jit
def _transpose_kernel(x, br, bc):
    m = x.shape[0]
    n = x.shape[1]
    dt = x.dtype
    out = nl.ndarray((n, m), dtype=dt, buffer=nl.shared_hbm)

    n_prog = nl.num_programs()
    pid = nl.program_id(0)
    ni = m // br
    nj = n // bc
    nb = ni * nj
    b_lo = (nb * pid) // n_prog
    b_hi = (nb * (pid + 1)) // n_prog
    nloc = b_hi - b_lo
    rt = br // P
    ct = bc // P

    # explicit SBUF input slots (prefetch ring)
    in0 = nl.ndarray((P, rt * bc), dtype=dt, buffer=nl.sbuf)
    in1 = nl.ndarray((P, rt * bc), dtype=dt, buffer=nl.sbuf)
    in2 = nl.ndarray((P, rt * bc), dtype=dt, buffer=nl.sbuf)

    # explicit SBUF output staging slots
    o0 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o1 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o2 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o3 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o4 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o5 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o6 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o7 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o8 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o9 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o10 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o11 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o12 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o13 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o14 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)
    o15 = nl.ndarray((P, br), dtype=dt, buffer=nl.sbuf)

    # prologue: prefetch first two blocks
    if nloc > 0:
        _load_block(x, in0, b_lo, ni, br, bc)
    if nloc > 1:
        _load_block(x, in1, b_lo + 1, ni, br, bc)

    for li in range(nloc):
        if li + 2 < nloc:
            _load_block(x, _sel3((li + 2) % 3, in0, in1, in2),
                        b_lo + li + 2, ni, br, bc)
        xin = _sel3(li % 3, in0, in1, in2)
        b = b_lo + li
        r0 = (b % ni) * br
        c0 = (b // ni) * bc
        if ct % 4 == 0:
            for gq in range(ct // 4):
                j0 = gq * 4
                g0 = li * ct + j0
                pa = nl.ndarray((P, br), dtype=dt, buffer=nl.psum)
                pb = nl.ndarray((P, br), dtype=dt, buffer=nl.psum)
                pc = nl.ndarray((P, br), dtype=dt, buffer=nl.psum)
                pd = nl.ndarray((P, br), dtype=dt, buffer=nl.psum)
                _transpose_into(pa, xin, j0, rt, bc)
                _transpose_into(pb, xin, j0 + 1, rt, bc)
                _transpose_into(pc, xin, j0 + 2, rt, bc)
                _transpose_into(pd, xin, j0 + 3, rt, bc)
                _evict_store(out, pa,
                             _pick_out(g0, o0, o1, o2, o3, o4, o5, o6, o7,
                                       o8, o9, o10, o11, o12, o13, o14, o15),
                             True, c0 + j0 * P, r0, br)
                _evict_store(out, pb,
                             _pick_out(g0 + 1, o0, o1, o2, o3, o4, o5, o6, o7,
                                       o8, o9, o10, o11, o12, o13, o14, o15),
                             False, c0 + (j0 + 1) * P, r0, br)
                _evict_store(out, pc,
                             _pick_out(g0 + 2, o0, o1, o2, o3, o4, o5, o6, o7,
                                       o8, o9, o10, o11, o12, o13, o14, o15),
                             True, c0 + (j0 + 2) * P, r0, br)
                _evict_store(out, pd,
                             _pick_out(g0 + 3, o0, o1, o2, o3, o4, o5, o6, o7,
                                       o8, o9, o10, o11, o12, o13, o14, o15),
                             False, c0 + (j0 + 3) * P, r0, br)
        else:
            for jc in range(ct):
                g = li * ct + jc
                ps = nl.ndarray((P, br), dtype=dt, buffer=nl.psum)
                _transpose_into(ps, xin, jc, rt, bc)
                _evict_store(out, ps,
                             _pick_out(g, o0, o1, o2, o3, o4, o5, o6, o7,
                                       o8, o9, o10, o11, o12, o13, o14, o15),
                             g % 2 == 0, c0 + jc * P, r0, br)
    return out


def run(x: torch.Tensor):
    global _LAST_CONFIG
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    m = int(x.shape[0])
    n = int(x.shape[1])
    br = _pick_block(m)
    bc = _pick_block(n)
    out = _transpose_kernel(x, br, bc)
    _LAST_CONFIG = {
        "block_rows": br,
        "block_cols": bc,
        "transpose_tile": P,
        "transpose_engine": "tensor (nc_transpose, fp16 psum)",
        "in_slots": _IN_SLOTS,
        "prefetch_blocks": _IN_SLOTS - 1,
        "psum_group": _PSUM_GROUP if (bc // P) % 4 == 0 else 1,
        "out_slots": _OUT_SLOTS,
        "evict_engines": "vector/scalar alternating",
        "core_split": "contiguous block ranges per program (LNC=2)",
    }
    return out


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
