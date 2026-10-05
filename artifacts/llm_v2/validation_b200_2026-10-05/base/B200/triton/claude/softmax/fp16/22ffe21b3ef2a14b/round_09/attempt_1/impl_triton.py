import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration (literals)
# ---------------------------------------------------------------------------
_BLOCK = 1024          # tile width along the row: 128 threads x 8 fp16 (16 B) per thread
_NUM_CHUNKS = 10       # tiles held on chip per row: 10 * 1024 = 10240 columns (exact fit)
_NUM_WARPS = 4         # 128 threads per row-program (cheap 4-warp reductions)
_NUM_STAGES = 1        # no runtime loop, nothing to pipeline
_MAXNREG = 96          # 5 row-programs resident per SM (5 * 128 * 96 <= 64K regs)
_ROWS_PER_PROGRAM = 1

_CONFIG = {
    "BLOCK": _BLOCK,
    "NUM_CHUNKS": _NUM_CHUNKS,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "maxnreg": _MAXNREG,
    "rows_per_program": _ROWS_PER_PROGRAM,
    "row_held_on_chip": True,
    "exp_formulation": "exp2_fma",
    "normalise": "reciprocal_multiply",
    "load_eviction": "evict_first",
}


@triton.jit
def _softmax_rows_kernel(x_ptr, y_ptr,
                         N_COLS: tl.constexpr,
                         BLOCK: tl.constexpr):
    LOG2E: tl.constexpr = 1.4426950408889634
    row = tl.program_id(0)
    base = row * N_COLS
    offs = tl.arange(0, BLOCK)
    xp = x_ptr + base + offs
    yp = y_ptr + base + offs

    # ---- single read of the whole row (10 tiles), upcast to fp32 ----------
    x0 = tl.load(xp + 0 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x1 = tl.load(xp + 1 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x2 = tl.load(xp + 2 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x3 = tl.load(xp + 3 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x4 = tl.load(xp + 4 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x5 = tl.load(xp + 5 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x6 = tl.load(xp + 6 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x7 = tl.load(xp + 7 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x8 = tl.load(xp + 8 * BLOCK, eviction_policy="evict_first").to(tl.float32)
    x9 = tl.load(xp + 9 * BLOCK, eviction_policy="evict_first").to(tl.float32)

    # ---- stage 1: row statistics (fp32) --------------------------------
    # The whole row is held on chip as one chunk: the online recurrence from
    # (m = -inf, l = 0) reduces to m = max(row), l = sum(exp(row - m)).
    xa = tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3))
    xb = tl.maximum(tl.maximum(x4, x5), tl.maximum(x6, x7))
    xm = tl.maximum(tl.maximum(xa, xb), tl.maximum(x8, x9))
    m = tl.max(xm, axis=0)                      # exact row maximum
    ms = m * LOG2E
    e0 = tl.exp2(x0 * LOG2E - ms)               # exp(x - m), fp32
    e1 = tl.exp2(x1 * LOG2E - ms)
    e2 = tl.exp2(x2 * LOG2E - ms)
    e3 = tl.exp2(x3 * LOG2E - ms)
    e4 = tl.exp2(x4 * LOG2E - ms)
    e5 = tl.exp2(x5 * LOG2E - ms)
    e6 = tl.exp2(x6 * LOG2E - ms)
    e7 = tl.exp2(x7 * LOG2E - ms)
    e8 = tl.exp2(x8 * LOG2E - ms)
    e9 = tl.exp2(x9 * LOG2E - ms)
    sa = (e0 + e1) + (e2 + e3)
    sb = (e4 + e5) + (e6 + e7)
    l = tl.sum((sa + sb) + (e8 + e9), axis=0)   # fp32 denominator

    # ---- stage 2: normalise from on-chip values, single downcast at store --
    inv = 1.0 / l
    out_ty = y_ptr.dtype.element_ty
    tl.store(yp + 0 * BLOCK, (e0 * inv).to(out_ty))
    tl.store(yp + 1 * BLOCK, (e1 * inv).to(out_ty))
    tl.store(yp + 2 * BLOCK, (e2 * inv).to(out_ty))
    tl.store(yp + 3 * BLOCK, (e3 * inv).to(out_ty))
    tl.store(yp + 4 * BLOCK, (e4 * inv).to(out_ty))
    tl.store(yp + 5 * BLOCK, (e5 * inv).to(out_ty))
    tl.store(yp + 6 * BLOCK, (e6 * inv).to(out_ty))
    tl.store(yp + 7 * BLOCK, (e7 * inv).to(out_ty))
    tl.store(yp + 8 * BLOCK, (e8 * inv).to(out_ty))
    tl.store(yp + 9 * BLOCK, (e9 * inv).to(out_ty))


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    x = x.contiguous()  # no-op guard for the contiguous input
    n_rows, n_cols = x.shape
    assert n_cols == _NUM_CHUNKS * _BLOCK, "kernel is specialised for an exact 10-tile row"
    assert n_rows * n_cols < 2 ** 31, "int32 offsets"
    y = torch.empty_like(x)
    grid = (n_rows,)
    _softmax_rows_kernel[grid](
        x, y,
        N_COLS=n_cols,
        BLOCK=_BLOCK,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        maxnreg=_MAXNREG,
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
