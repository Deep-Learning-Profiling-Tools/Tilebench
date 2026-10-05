import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration (literals)
# ---------------------------------------------------------------------------
_BLOCK = 2048          # tile width along the row (elements)
_NUM_CHUNKS = 5        # tiles held on chip per row: 5 * 2048 = 10240 columns
_NUM_WARPS = 4         # 128 threads -> 2 x 16 B vectors per thread per tile, 80 elems/thread
_NUM_STAGES = 1        # no runtime loop, nothing to pipeline
_MAXNREG = 96          # 128 thr * 96 regs -> 5 CTAs (rows) resident per SM
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
}


@triton.jit
def _softmax_rows_kernel(x_ptr, y_ptr,
                         N_COLS: tl.constexpr,
                         BLOCK: tl.constexpr,
                         MASKED: tl.constexpr):
    LOG2E: tl.constexpr = 1.4426950408889634
    row = tl.program_id(0)
    base = row * N_COLS
    offs = tl.arange(0, BLOCK)
    xp = x_ptr + base + offs
    yp = y_ptr + base + offs

    # ---- single read of the whole row, held on chip as 5 tiles (fp32) ----
    if MASKED:
        k0 = (offs + 0 * BLOCK) < N_COLS
        k1 = (offs + 1 * BLOCK) < N_COLS
        k2 = (offs + 2 * BLOCK) < N_COLS
        k3 = (offs + 3 * BLOCK) < N_COLS
        k4 = (offs + 4 * BLOCK) < N_COLS
        x0 = tl.load(xp + 0 * BLOCK, mask=k0, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)
        x1 = tl.load(xp + 1 * BLOCK, mask=k1, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)
        x2 = tl.load(xp + 2 * BLOCK, mask=k2, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)
        x3 = tl.load(xp + 3 * BLOCK, mask=k3, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)
        x4 = tl.load(xp + 4 * BLOCK, mask=k4, other=-float("inf"), eviction_policy="evict_first").to(tl.float32)
    else:
        x0 = tl.load(xp + 0 * BLOCK, eviction_policy="evict_first").to(tl.float32)
        x1 = tl.load(xp + 1 * BLOCK, eviction_policy="evict_first").to(tl.float32)
        x2 = tl.load(xp + 2 * BLOCK, eviction_policy="evict_first").to(tl.float32)
        x3 = tl.load(xp + 3 * BLOCK, eviction_policy="evict_first").to(tl.float32)
        x4 = tl.load(xp + 4 * BLOCK, eviction_policy="evict_first").to(tl.float32)

    # ---- stage 1: row statistics (fp32) --------------------------------
    # The whole row is held on chip as one chunk: the online recurrence with
    # initial state (m = -inf, l = 0) reduces to m = max(row), l = sum(exp(row - m)).
    xm = tl.maximum(tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3)), x4)
    m = tl.max(xm, axis=0)                      # exact row maximum (fp32)
    ms = m * LOG2E
    e0 = tl.exp2(x0 * LOG2E - ms)               # exp(x - m), fp32
    e1 = tl.exp2(x1 * LOG2E - ms)
    e2 = tl.exp2(x2 * LOG2E - ms)
    e3 = tl.exp2(x3 * LOG2E - ms)
    e4 = tl.exp2(x4 * LOG2E - ms)
    l = tl.sum(((e0 + e1) + (e2 + e3)) + e4, axis=0)   # fp32 denominator

    # ---- stage 2: normalise from on-chip values, single downcast at store --
    inv = 1.0 / l
    out_ty = y_ptr.dtype.element_ty
    if MASKED:
        tl.store(yp + 0 * BLOCK, (e0 * inv).to(out_ty), mask=k0)
        tl.store(yp + 1 * BLOCK, (e1 * inv).to(out_ty), mask=k1)
        tl.store(yp + 2 * BLOCK, (e2 * inv).to(out_ty), mask=k2)
        tl.store(yp + 3 * BLOCK, (e3 * inv).to(out_ty), mask=k3)
        tl.store(yp + 4 * BLOCK, (e4 * inv).to(out_ty), mask=k4)
    else:
        tl.store(yp + 0 * BLOCK, (e0 * inv).to(out_ty))
        tl.store(yp + 1 * BLOCK, (e1 * inv).to(out_ty))
        tl.store(yp + 2 * BLOCK, (e2 * inv).to(out_ty))
        tl.store(yp + 3 * BLOCK, (e3 * inv).to(out_ty))
        tl.store(yp + 4 * BLOCK, (e4 * inv).to(out_ty))


def run(x):
    assert x.dim() == 2, "softmax expects a 2-D input"
    x = x.contiguous()  # no-op guard for the contiguous input
    n_rows, n_cols = x.shape
    assert 0 < n_cols <= _NUM_CHUNKS * _BLOCK, "row must fit in the on-chip tiles"
    assert n_rows * n_cols < 2 ** 31, "int32 offsets"
    y = torch.empty_like(x)
    masked = (n_cols != _NUM_CHUNKS * _BLOCK)
    grid = (n_rows,)
    _softmax_rows_kernel[grid](
        x, y,
        N_COLS=n_cols,
        BLOCK=_BLOCK,
        MASKED=masked,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        maxnreg=_MAXNREG,
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
