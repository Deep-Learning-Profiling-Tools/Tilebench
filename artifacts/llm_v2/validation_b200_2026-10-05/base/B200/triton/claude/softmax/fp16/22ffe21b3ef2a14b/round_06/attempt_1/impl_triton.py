import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ---------------------------------------------------------------------------
# Fixed configuration (literals)
# ---------------------------------------------------------------------------
_BLOCK = 2048          # tile width along the row (elements)
_NUM_CHUNKS = 5        # tiles held on chip per row: 5 * 2048 = 10240 columns
_NUM_WARPS = 8         # 256 threads -> 8 fp16 (16 B) per thread per tile
_LOOP_STAGES = 3       # software-pipeline depth of the persistent row loop (cp.async prefetch)
_CTAS_PER_SM = 3       # resident programs per SM (guaranteed by _MAXNREG)
_NUM_SMS = 148         # B200 SM count
_MAXNREG = 80          # 256 thr * 80 regs * 3 CTAs = 61440 <= 65536 regs/SM

_CONFIG = {
    "BLOCK": _BLOCK,
    "NUM_CHUNKS": _NUM_CHUNKS,
    "num_warps": _NUM_WARPS,
    "num_stages": _LOOP_STAGES,
    "loop_num_stages": _LOOP_STAGES,
    "maxnreg": _MAXNREG,
    "ctas_per_sm": _CTAS_PER_SM,
    "num_sms": _NUM_SMS,
    "persistent_programs": _NUM_SMS * _CTAS_PER_SM,
    "row_held_on_chip": True,
    "exp_formulation": "libdevice_exp2_fma",
    "normalise": "reciprocal_multiply",
}


@triton.jit
def _softmax_persistent_kernel(x_ptr, y_ptr,
                               N_ROWS: tl.constexpr,
                               N_COLS: tl.constexpr,
                               BLOCK: tl.constexpr,
                               NPROG: tl.constexpr,
                               LOOP_STAGES: tl.constexpr,
                               MASKED: tl.constexpr):
    LOG2E: tl.constexpr = 1.4426950408889634
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    out_ty = y_ptr.dtype.element_ty

    if MASKED:
        k0 = (offs + 0 * BLOCK) < N_COLS
        k1 = (offs + 1 * BLOCK) < N_COLS
        k2 = (offs + 2 * BLOCK) < N_COLS
        k3 = (offs + 3 * BLOCK) < N_COLS
        k4 = (offs + 4 * BLOCK) < N_COLS

    # Persistent grid-stride loop over rows; loads of upcoming rows are
    # prefetched (cp.async) while the current row is reduced and stored.
    for row in tl.range(pid, N_ROWS, NPROG, num_stages=LOOP_STAGES):
        base = row * N_COLS
        xp = x_ptr + base + offs
        yp = y_ptr + base + offs

        # ---- single read of the whole row, held on chip as 5 tiles (fp32) ----
        if MASKED:
            x0 = tl.load(xp + 0 * BLOCK, mask=k0, other=-float("inf")).to(tl.float32)
            x1 = tl.load(xp + 1 * BLOCK, mask=k1, other=-float("inf")).to(tl.float32)
            x2 = tl.load(xp + 2 * BLOCK, mask=k2, other=-float("inf")).to(tl.float32)
            x3 = tl.load(xp + 3 * BLOCK, mask=k3, other=-float("inf")).to(tl.float32)
            x4 = tl.load(xp + 4 * BLOCK, mask=k4, other=-float("inf")).to(tl.float32)
        else:
            x0 = tl.load(xp + 0 * BLOCK).to(tl.float32)
            x1 = tl.load(xp + 1 * BLOCK).to(tl.float32)
            x2 = tl.load(xp + 2 * BLOCK).to(tl.float32)
            x3 = tl.load(xp + 3 * BLOCK).to(tl.float32)
            x4 = tl.load(xp + 4 * BLOCK).to(tl.float32)

        # ---- stage 1: row statistics (fp32) --------------------------------
        # Whole row held on chip as one chunk: the online recurrence from
        # (m = -inf, l = 0) reduces to m = max(row), l = sum(exp(row - m)).
        xm = tl.maximum(tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3)), x4)
        m = tl.max(xm, axis=0)                      # exact row maximum (fp32)
        ms = m * LOG2E
        e0 = libdevice.exp2(x0 * LOG2E - ms)        # exp(x - m), fp32
        e1 = libdevice.exp2(x1 * LOG2E - ms)
        e2 = libdevice.exp2(x2 * LOG2E - ms)
        e3 = libdevice.exp2(x3 * LOG2E - ms)
        e4 = libdevice.exp2(x4 * LOG2E - ms)
        l = tl.sum(((e0 + e1) + (e2 + e3)) + e4, axis=0)   # fp32 denominator

        # ---- stage 2: normalise from on-chip values, single downcast at store --
        inv = 1.0 / l
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
    nprog = min(n_rows, _NUM_SMS * _CTAS_PER_SM)
    masked = (n_cols != _NUM_CHUNKS * _BLOCK)
    _softmax_persistent_kernel[(nprog,)](
        x, y,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        BLOCK=_BLOCK,
        NPROG=nprog,
        LOOP_STAGES=_LOOP_STAGES,
        MASKED=masked,
        num_warps=_NUM_WARPS,
        num_stages=_LOOP_STAGES,
        maxnreg=_MAXNREG,
    )
    return y


def get_last_config() -> dict:
    return dict(_CONFIG)
