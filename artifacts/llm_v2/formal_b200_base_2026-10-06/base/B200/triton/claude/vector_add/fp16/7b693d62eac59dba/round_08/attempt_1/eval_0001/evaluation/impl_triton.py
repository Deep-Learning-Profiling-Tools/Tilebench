import torch
import triton
import triton.language as tl

# Fixed configuration literals
_TILE = 2048          # elements per pipeline step of a program (4 KB of fp16 per tensor)
_PROGRAMS = 640       # target grid: 10240 tiles / 640 = 16 tiles per program (exact balance)
_DEPTH = 4            # register software-pipeline depth (tiles in flight per program)
_NUM_WARPS = 4
_NUM_STAGES = 1
_MAXNREG = 96         # 128 thr x 96 regs -> 5 CTAs/SM -> 740 slots >= 640 (single resident wave)

_CONFIG = {
    "mode": "grid_stride_sweep_single_wave",
    "TILE": _TILE,
    "PROGRAMS": _PROGRAMS,
    "pipeline_depth": _DEPTH,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "maxnreg": _MAXNREG,
    "load_eviction_policy": "evict_first",
}


@triton.jit
def _ld2(x_ptr, y_ptr, offs, n, EVEN: tl.constexpr):
    if EVEN:
        xv = tl.load(x_ptr + offs, eviction_policy="evict_first")
        yv = tl.load(y_ptr + offs, eviction_policy="evict_first")
    else:
        m = offs < n
        xv = tl.load(x_ptr + offs, mask=m, other=0.0, eviction_policy="evict_first")
        yv = tl.load(y_ptr + offs, mask=m, other=0.0, eviction_policy="evict_first")
    return xv, yv


@triton.jit
def _st(o_ptr, offs, v, n, EVEN: tl.constexpr):
    if EVEN:
        tl.store(o_ptr + offs, v)
    else:
        tl.store(o_ptr + offs, v, mask=offs < n)


@triton.jit
def _vadd_sweep(x_ptr, y_ptr, o_ptr, n,
                TILE: tl.constexpr, TPP: tl.constexpr, G: tl.constexpr,
                EVEN: tl.constexpr, USE_I64: tl.constexpr):
    pid = tl.program_id(0)
    if USE_I64:
        pid = pid.to(tl.int64)
    ar = tl.arange(0, TILE)
    # Grid-stride sweep: program p handles tiles p, p+G, ..., p+(TPP-1)*G.
    o0 = pid * TILE + ar
    x0, y0 = _ld2(x_ptr, y_ptr, o0, n, EVEN)
    o1 = o0 + G * TILE
    x1, y1 = _ld2(x_ptr, y_ptr, o1, n, EVEN)
    o2 = o1 + G * TILE
    x2, y2 = _ld2(x_ptr, y_ptr, o2, n, EVEN)

    # Steady state: issue loads for step i+3, then add/store step i.
    for _i in tl.static_range(TPP - 3):
        o3 = o2 + G * TILE
        x3, y3 = _ld2(x_ptr, y_ptr, o3, n, EVEN)
        _st(o_ptr, o0, x0 + y0, n, EVEN)
        o0 = o1
        x0 = x1
        y0 = y1
        o1 = o2
        x1 = x2
        y1 = y2
        o2 = o3
        x2 = x3
        y2 = y3

    _st(o_ptr, o0, x0 + y0, n, EVEN)
    _st(o_ptr, o1, x1 + y1, n, EVEN)
    _st(o_ptr, o2, x2 + y2, n, EVEN)


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    tiles = triton.cdiv(n, _TILE)
    tpp = max(triton.cdiv(tiles, _PROGRAMS), _DEPTH - 1)   # tiles per program (>= 3)
    grid = triton.cdiv(tiles, tpp)                          # programs actually launched
    even = (n % _TILE == 0) and (grid * tpp == tiles)
    use_i64 = (grid * tpp * _TILE) >= 2 ** 31
    _vadd_sweep[(grid,)](
        x, y, out, n,
        TILE=_TILE, TPP=tpp, G=grid, EVEN=even, USE_I64=use_i64,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES, maxnreg=_MAXNREG,
    )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
