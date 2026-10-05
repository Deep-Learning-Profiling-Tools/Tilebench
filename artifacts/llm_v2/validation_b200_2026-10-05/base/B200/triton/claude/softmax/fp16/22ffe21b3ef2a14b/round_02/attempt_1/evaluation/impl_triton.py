import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration (literals)
# ---------------------------------------------------------------------------
_BLOCK = 2048            # tile width along the row (elements)
_NUM_CHUNKS = 5          # tiles held on chip per row: 5 * 2048 = 10240 columns
_NUM_WARPS = 8           # 256 threads -> 8 fp16 (16 B) per thread per tile
_NUM_STAGES = 1          # no compiler pipelining; prefetch is explicit
_NUM_SMS = 148           # B200 SM count (fixed device)
_PROGRAMS_PER_SM = 2     # persistent programs per SM
_PREFETCH_DEPTH = 2      # rows prefetched ahead (raw fp16 in registers)
_MAXNREG = 128           # guarantees 2 x 256-thread programs resident per SM

_CONFIG = {
    "BLOCK": _BLOCK,
    "NUM_CHUNKS": _NUM_CHUNKS,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "num_sms": _NUM_SMS,
    "programs_per_sm": _PROGRAMS_PER_SM,
    "prefetch_depth": _PREFETCH_DEPTH,
    "maxnreg": _MAXNREG,
    "persistent": True,
    "row_held_on_chip": True,
    "exp_formulation": "exp2_fma",
    "normalise": "reciprocal_multiply",
}


@triton.jit
def _ld(ptrs, col_ok, pf, MASKED: tl.constexpr):
    if MASKED:
        v = tl.load(ptrs, mask=col_ok & pf, other=-float("inf"),
                    eviction_policy="evict_first")
    else:
        v = tl.load(ptrs, mask=pf, other=0.0, eviction_policy="evict_first")
    return v


@triton.jit
def _st(ptrs, val, col_ok, MASKED: tl.constexpr):
    if MASKED:
        tl.store(ptrs, val, mask=col_ok)
    else:
        tl.store(ptrs, val)


@triton.jit
def _softmax_persistent_kernel(x_ptr, y_ptr, N_ROWS,
                               N_COLS: tl.constexpr,
                               BLOCK: tl.constexpr,
                               NPROG: tl.constexpr,
                               MASKED: tl.constexpr):
    LOG2E: tl.constexpr = 1.4426950408889634
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    c0 = (offs + 0 * BLOCK) < N_COLS
    c1 = (offs + 1 * BLOCK) < N_COLS
    c2 = (offs + 2 * BLOCK) < N_COLS
    c3 = (offs + 3 * BLOCK) < N_COLS
    c4 = (offs + 4 * BLOCK) < N_COLS

    # ---- prologue: prefetch rows pid and pid + NPROG (raw fp16) ----------
    r_a = pid
    ok_a = r_a < N_ROWS
    pa = x_ptr + r_a * N_COLS + offs
    a0 = _ld(pa + 0 * BLOCK, c0, ok_a, MASKED)
    a1 = _ld(pa + 1 * BLOCK, c1, ok_a, MASKED)
    a2 = _ld(pa + 2 * BLOCK, c2, ok_a, MASKED)
    a3 = _ld(pa + 3 * BLOCK, c3, ok_a, MASKED)
    a4 = _ld(pa + 4 * BLOCK, c4, ok_a, MASKED)

    r_b = pid + NPROG
    ok_b = r_b < N_ROWS
    pb = x_ptr + r_b * N_COLS + offs
    b0 = _ld(pb + 0 * BLOCK, c0, ok_b, MASKED)
    b1 = _ld(pb + 1 * BLOCK, c1, ok_b, MASKED)
    b2 = _ld(pb + 2 * BLOCK, c2, ok_b, MASKED)
    b3 = _ld(pb + 3 * BLOCK, c3, ok_b, MASKED)
    b4 = _ld(pb + 4 * BLOCK, c4, ok_b, MASKED)

    out_ty = y_ptr.dtype.element_ty
    for row in range(pid, N_ROWS, NPROG):
        # upcast the current row (single read of the row, held on chip)
        x0 = a0.to(tl.float32)
        x1 = a1.to(tl.float32)
        x2 = a2.to(tl.float32)
        x3 = a3.to(tl.float32)
        x4 = a4.to(tl.float32)

        # rotate prefetch buffers and issue the load two rows ahead
        a0 = b0
        a1 = b1
        a2 = b2
        a3 = b3
        a4 = b4
        nxt = row + 2 * NPROG
        pf = nxt < N_ROWS
        pn = x_ptr + nxt * N_COLS + offs
        b0 = _ld(pn + 0 * BLOCK, c0, pf, MASKED)
        b1 = _ld(pn + 1 * BLOCK, c1, pf, MASKED)
        b2 = _ld(pn + 2 * BLOCK, c2, pf, MASKED)
        b3 = _ld(pn + 3 * BLOCK, c3, pf, MASKED)
        b4 = _ld(pn + 4 * BLOCK, c4, pf, MASKED)

        # ---- stage 1: row statistics (fp32) ----------------------------
        # Whole row held on chip as one chunk: the online recurrence from
        # (m=-inf, l=0) reduces to m = max(row), l = sum(exp(row - m)).
        xm = tl.maximum(tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3)), x4)
        m = tl.max(xm, axis=0)                  # exact row maximum
        ms = m * LOG2E
        e0 = tl.exp2(x0 * LOG2E - ms)           # exp(x - m), fp32
        e1 = tl.exp2(x1 * LOG2E - ms)
        e2 = tl.exp2(x2 * LOG2E - ms)
        e3 = tl.exp2(x3 * LOG2E - ms)
        e4 = tl.exp2(x4 * LOG2E - ms)
        l = tl.sum(((e0 + e1) + (e2 + e3)) + e4, axis=0)   # fp32 denominator

        # ---- stage 2: normalise from on-chip values, single downcast ----
        inv = 1.0 / l
        py = y_ptr + row * N_COLS + offs
        _st(py + 0 * BLOCK, (e0 * inv).to(out_ty), c0, MASKED)
        _st(py + 1 * BLOCK, (e1 * inv).to(out_ty), c1, MASKED)
        _st(py + 2 * BLOCK, (e2 * inv).to(out_ty), c2, MASKED)
        _st(py + 3 * BLOCK, (e3 * inv).to(out_ty), c3, MASKED)
        _st(py + 4 * BLOCK, (e4 * inv).to(out_ty), c4, MASKED)


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    x = x.contiguous()  # no-op guard for the contiguous input
    n_rows, n_cols = x.shape
    assert 0 < n_cols <= _NUM_CHUNKS * _BLOCK, "row must fit in the on-chip tiles"
    y = torch.empty_like(x)
    nprog = min(_NUM_SMS * _PROGRAMS_PER_SM, n_rows)
    masked = (n_cols != _NUM_CHUNKS * _BLOCK)
    _softmax_persistent_kernel[(nprog,)](
        x, y, n_rows,
        N_COLS=n_cols,
        BLOCK=_BLOCK,
        NPROG=nprog,
        MASKED=masked,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        maxnreg=_MAXNREG,
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
