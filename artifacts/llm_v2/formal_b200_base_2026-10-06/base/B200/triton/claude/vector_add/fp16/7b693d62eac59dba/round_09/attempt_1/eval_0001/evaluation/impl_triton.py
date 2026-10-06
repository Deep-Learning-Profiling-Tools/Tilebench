import torch
import triton
import triton.language as tl

# Fixed configuration literals
_TILE = 4096          # elements per tile (per pipeline step of a program); 8 KB fp16 per tensor
_PROGRAMS = 1024      # target grid size (grid-stride sweep over tiles)
_NUM_WARPS = 4
_NUM_STAGES = 1
_LOAD_EVICT = "evict_first"   # inputs are streamed once; make them preferred L2 victims

_CONFIG = {
    "TILE": _TILE,
    "PROGRAMS": _PROGRAMS,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "load_eviction_policy": _LOAD_EVICT,
    "pipeline_depth": 2,
    "offsets": "int32_unless_overflow",
}


@triton.jit
def _load_xy(x_ptr, y_ptr, offs, n, EVEN: tl.constexpr):
    if EVEN:
        xv = tl.load(x_ptr + offs, eviction_policy="evict_first")
        yv = tl.load(y_ptr + offs, eviction_policy="evict_first")
    else:
        m = offs < n
        xv = tl.load(x_ptr + offs, mask=m, other=0.0, eviction_policy="evict_first")
        yv = tl.load(y_ptr + offs, mask=m, other=0.0, eviction_policy="evict_first")
    return xv, yv


@triton.jit
def _store_sum(out_ptr, offs, xv, yv, n, EVEN: tl.constexpr):
    if EVEN:
        tl.store(out_ptr + offs, xv + yv)
    else:
        tl.store(out_ptr + offs, xv + yv, mask=offs < n)


@triton.jit
def _vector_add_pipelined(x_ptr, y_ptr, out_ptr, n,
                          TILE: tl.constexpr, TPP: tl.constexpr,
                          G: tl.constexpr, EVEN: tl.constexpr,
                          USE_I64: tl.constexpr):
    pid = tl.program_id(0)
    if USE_I64:
        pid = pid.to(tl.int64)
    ar = tl.arange(0, TILE)
    offs = pid * TILE + ar
    offs = tl.max_contiguous(tl.multiple_of(offs, TILE), TILE)
    xv, yv = _load_xy(x_ptr, y_ptr, offs, n, EVEN)
    # Software pipeline: issue loads of tile i before storing tile i-1.
    for i in tl.static_range(1, TPP):
        noffs = offs + G * TILE
        noffs = tl.max_contiguous(tl.multiple_of(noffs, TILE), TILE)
        nx, ny = _load_xy(x_ptr, y_ptr, noffs, n, EVEN)
        _store_sum(out_ptr, offs, xv, yv, n, EVEN)
        offs = noffs
        xv = nx
        yv = ny
    _store_sum(out_ptr, offs, xv, yv, n, EVEN)


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    tiles_total = triton.cdiv(n, _TILE)
    tpp = triton.cdiv(tiles_total, _PROGRAMS)          # tiles per program
    grid_size = triton.cdiv(tiles_total, tpp)          # programs actually launched
    # Grid-stride assignment: program p handles tiles p, p+G, ..., p+(tpp-1)*G.
    even = (n % _TILE == 0) and (grid_size * tpp == tiles_total)
    use_i64 = (grid_size * tpp * _TILE) >= 2 ** 31
    _vector_add_pipelined[(grid_size,)](
        x, y, out, n,
        TILE=_TILE, TPP=tpp, G=grid_size, EVEN=even, USE_I64=use_i64,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
