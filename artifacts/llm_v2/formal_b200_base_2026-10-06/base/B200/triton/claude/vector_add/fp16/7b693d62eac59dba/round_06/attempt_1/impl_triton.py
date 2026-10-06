import torch
import triton
import triton.language as tl

# Fixed configuration literals
_TILE = 4096          # elements per pipeline step of a program (8 KB per tensor)
_UNIT = 512           # partition granularity in elements (1 KB of fp16)
_NUM_SMS = 148        # B200 SM count
_CTAS_PER_SM = 4      # programs per SM -> grid = 592, all co-resident
_NUM_WARPS = 4
_NUM_STAGES = 1
_MAXNREG = 128        # 4 CTAs x 128 threads x 128 regs = 64K regs/SM (residency guarantee)
_DEPTH = 3            # register software-pipeline depth (tiles in flight per program)
_TAIL_BLOCK = 512     # masked tail kernel block (only used if n % _UNIT != 0)

_CONFIG = {
    "mode": "balanced_contiguous_chunks",
    "TILE": _TILE,
    "UNIT": _UNIT,
    "NUM_SMS": _NUM_SMS,
    "CTAS_PER_SM": _CTAS_PER_SM,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "maxnreg": _MAXNREG,
    "pipeline_depth": _DEPTH,
    "load_eviction_policy": "evict_first",
    "TAIL_BLOCK": _TAIL_BLOCK,
}


@triton.jit
def _ld2(x_ptr, y_ptr, offs, end, MASKED: tl.constexpr):
    if MASKED:
        m = offs < end
        xv = tl.load(x_ptr + offs, mask=m, eviction_policy="evict_first")
        yv = tl.load(y_ptr + offs, mask=m, eviction_policy="evict_first")
    else:
        xv = tl.load(x_ptr + offs, eviction_policy="evict_first")
        yv = tl.load(y_ptr + offs, eviction_policy="evict_first")
    return xv, yv


@triton.jit
def _st(o_ptr, offs, v, end, MASKED: tl.constexpr):
    if MASKED:
        tl.store(o_ptr + offs, v, mask=offs < end)
    else:
        tl.store(o_ptr + offs, v)


@triton.jit
def _vadd_balanced(x_ptr, y_ptr, o_ptr, q, r,
                   TILE: tl.constexpr, UNIT: tl.constexpr, STEPS: tl.constexpr,
                   MASK_ALL: tl.constexpr, USE_I64: tl.constexpr):
    pid = tl.program_id(0)
    if USE_I64:
        pid = pid.to(tl.int64)
    # Balanced contiguous partition in units of UNIT elements:
    # first r programs own q+1 units, the rest own q units.
    su = pid * q + tl.minimum(pid, r)
    cnt = q + (pid < r).to(su.dtype)
    start = su * UNIT
    end = start + cnt * UNIT
    ar = tl.arange(0, TILE)

    # Prologue: issue loads for steps 0 and 1.
    o0 = start + ar
    x0, y0 = _ld2(x_ptr, y_ptr, o0, end, MASK_ALL)
    o1 = o0 + TILE
    x1, y1 = _ld2(x_ptr, y_ptr, o1, end, MASK_ALL)

    # Steady state: load step i+2, then add/store step i.
    for _i in tl.static_range(STEPS - 3):
        o2 = o1 + TILE
        x2, y2 = _ld2(x_ptr, y_ptr, o2, end, MASK_ALL)
        _st(o_ptr, o0, x0 + y0, end, MASK_ALL)
        o0 = o1
        x0 = x1
        y0 = y1
        o1 = o2
        x1 = x2
        y1 = y2

    # Last step (partial for this shape) is always masked.
    o2 = o1 + TILE
    x2, y2 = _ld2(x_ptr, y_ptr, o2, end, True)
    _st(o_ptr, o0, x0 + y0, end, MASK_ALL)
    _st(o_ptr, o1, x1 + y1, end, MASK_ALL)
    _st(o_ptr, o2, x2 + y2, end, True)


@triton.jit
def _vadd_tail_kernel(x_ptr, y_ptr, o_ptr, start, n, BLOCK: tl.constexpr):
    offs = start + tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    xv = tl.load(x_ptr + offs, mask=m)
    yv = tl.load(y_ptr + offs, mask=m)
    tl.store(o_ptr + offs, xv + yv, mask=m)


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out

    units = n // _UNIT
    if units > 0:
        grid = min(_NUM_SMS * _CTAS_PER_SM, units)
        q, r = divmod(units, grid)
        max_chunk = (q + (1 if r else 0)) * _UNIT
        min_chunk = q * _UNIT
        steps = max(triton.cdiv(max_chunk, _TILE), _DEPTH)
        # Unmasked steps 0..steps-2 are valid only if every chunk covers them fully.
        mask_all = not (min_chunk >= (steps - 1) * _TILE)
        use_i64 = (n + steps * _TILE) >= 2 ** 31
        _vadd_balanced[(grid,)](
            x, y, out, q, r,
            TILE=_TILE, UNIT=_UNIT, STEPS=steps,
            MASK_ALL=mask_all, USE_I64=use_i64,
            num_warps=_NUM_WARPS, num_stages=_NUM_STAGES, maxnreg=_MAXNREG,
        )

    start = units * _UNIT
    rem = n - start
    if rem > 0:
        _vadd_tail_kernel[(triton.cdiv(rem, _TAIL_BLOCK),)](
            x, y, out, start, n,
            BLOCK=_TAIL_BLOCK, num_warps=4,
        )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
