import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration (literals)
# ---------------------------------------------------------------------------
_ROWS_T = 256                 # tile rows: one 16-byte vector per thread per chunk
_VEC = 8                      # fp16 elements per thread per chunk (16 B)
_CHUNK = _ROWS_T * _VEC       # 2048 columns per chunk
_NUM_CHUNKS = 5               # 5 * 2048 = 10240 columns (exact fit)
_NUM_WARPS = 8                # 256 threads, layout [32,1] x [8,1] over the [256, 8] tile
_NUM_STAGES = 1               # no runtime loop
_MAXNREG = 40                 # ~6 CTAs (rows) resident per SM
_ROWS_PER_PROGRAM = 1

_CONFIG = {
    "ROWS_T": _ROWS_T,
    "VEC": _VEC,
    "CHUNK": _CHUNK,
    "NUM_CHUNKS": _NUM_CHUNKS,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "maxnreg": _MAXNREG,
    "rows_per_program": _ROWS_PER_PROGRAM,
    "sweep1": "row held on chip (fp16 regs), per-thread online (m,l) over chunks + rescaling merge",
    "sweep2": "re-read row (L1/L2), evict_first",
    "exp_formulation": "exp2_fma",
    "normalise": "reciprocal_multiply",
}


@triton.jit
def _online_update(m_t, l_t, f):
    # Online rescaling recurrence on one chunk (per thread-owned slice of the tile):
    #   m_new = max(m, m_c);  l = l * exp(m - m_new) + sum exp(chunk - m_new);  m = m_new
    LOG2E: tl.constexpr = 1.4426950408889634
    mc = tl.max(f, axis=1)                         # intra-thread (layout [1, 8] per thread)
    mn = tl.maximum(m_t, mc)
    mnL = mn * LOG2E
    p = tl.exp2(f * LOG2E - mnL[:, None])          # exp(x - m_new), fp32
    l_t = l_t * tl.exp2((m_t - mn) * LOG2E) + tl.sum(p, axis=1)
    return mn, l_t


@triton.jit
def _softmax_rows_kernel(x_ptr, y_ptr,
                         N_COLS: tl.constexpr,
                         ROWS_T: tl.constexpr,
                         VEC: tl.constexpr):
    LOG2E: tl.constexpr = 1.4426950408889634
    CHUNK: tl.constexpr = ROWS_T * VEC
    row = tl.program_id(0)
    base = row * N_COLS
    r = tl.arange(0, ROWS_T)
    v = tl.arange(0, VEC)
    offs = base + r[:, None] * VEC + v[None, :]    # [256, 8], 16 B contiguous per thread
    xp = x_ptr + offs
    yp = y_ptr + offs

    # ---- sweep 1: read the whole row once (held on chip as packed fp16) ----
    a0 = tl.load(xp + 0 * CHUNK)
    a1 = tl.load(xp + 1 * CHUNK)
    a2 = tl.load(xp + 2 * CHUNK)
    a3 = tl.load(xp + 3 * CHUNK)
    a4 = tl.load(xp + 4 * CHUNK)

    # Row statistics in fp32: online recurrence from (m = -inf, l = 0), every
    # element upcast to fp32 before use.
    m_t = tl.full([ROWS_T], float("-inf"), tl.float32)
    l_t = tl.zeros([ROWS_T], tl.float32)
    m_t, l_t = _online_update(m_t, l_t, a0.to(tl.float32))
    m_t, l_t = _online_update(m_t, l_t, a1.to(tl.float32))
    m_t, l_t = _online_update(m_t, l_t, a2.to(tl.float32))
    m_t, l_t = _online_update(m_t, l_t, a3.to(tl.float32))
    m_t, l_t = _online_update(m_t, l_t, a4.to(tl.float32))

    # Merge the partial online states with the same rescaling rule.
    m = tl.max(m_t, axis=0)                                    # exact row maximum
    l = tl.sum(l_t * tl.exp2((m_t - m) * LOG2E), axis=0)       # fp32 denominator
    inv = 1.0 / l
    mL = m * LOG2E

    # ---- sweep 2: re-read the row (L1/L2 hit) and normalise ----------------
    b0 = tl.load(xp + 0 * CHUNK, eviction_policy="evict_first")
    b1 = tl.load(xp + 1 * CHUNK, eviction_policy="evict_first")
    b2 = tl.load(xp + 2 * CHUNK, eviction_policy="evict_first")
    b3 = tl.load(xp + 3 * CHUNK, eviction_policy="evict_first")
    b4 = tl.load(xp + 4 * CHUNK, eviction_policy="evict_first")

    out_ty = y_ptr.dtype.element_ty
    tl.store(yp + 0 * CHUNK, (tl.exp2(b0.to(tl.float32) * LOG2E - mL) * inv).to(out_ty))
    tl.store(yp + 1 * CHUNK, (tl.exp2(b1.to(tl.float32) * LOG2E - mL) * inv).to(out_ty))
    tl.store(yp + 2 * CHUNK, (tl.exp2(b2.to(tl.float32) * LOG2E - mL) * inv).to(out_ty))
    tl.store(yp + 3 * CHUNK, (tl.exp2(b3.to(tl.float32) * LOG2E - mL) * inv).to(out_ty))
    tl.store(yp + 4 * CHUNK, (tl.exp2(b4.to(tl.float32) * LOG2E - mL) * inv).to(out_ty))


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    x = x.contiguous()  # no-op guard for the contiguous input
    n_rows, n_cols = x.shape
    assert n_cols == _NUM_CHUNKS * _CHUNK, "kernel is specialised for an exact 5-chunk row"
    assert n_rows * n_cols < 2 ** 31, "int32 offsets"
    y = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_rows_kernel[grid](
        x, y,
        N_COLS=n_cols,
        ROWS_T=_ROWS_T,
        VEC=_VEC,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        maxnreg=_MAXNREG,
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
