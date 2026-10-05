import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration literals
# ---------------------------------------------------------------------------
# Persistent grid-stride kernel: each program owns tiles pid, pid+G, pid+2G...
# and keeps the NEXT tile's loads in flight (register double-buffering) while
# it adds/stores the current one.  The grid is capped at NUM_SMS*CTAS_PER_SM
# programs so every program is co-resident (maxnreg caps registers so that
# 4 x 128-thread CTAs always fit on an SM); no second wave, no CTA turnover.
_BLOCK = 4096          # elements per tile (32 fp16 / thread / tensor)
_NUM_WARPS = 4
_NUM_STAGES = 1
_NUM_SMS = 148
_CTAS_PER_SM = 4
_MAXNREG = 128

_CONFIG = {
    "BLOCK": _BLOCK,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "num_sms": _NUM_SMS,
    "ctas_per_sm": _CTAS_PER_SM,
    "maxnreg": _MAXNREG,
    "schedule": "persistent_grid_stride_register_prefetch",
    "load_eviction_policy": "evict_first",
    "store_cache_modifier": "default",
}


@triton.jit
def _tile_offs(tile, BLOCK: tl.constexpr, I64: tl.constexpr):
    ar = tl.arange(0, BLOCK)
    if I64:
        base = tile.to(tl.int64) * BLOCK
    else:
        base = tile * BLOCK
    offs = base + ar
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    return offs


@triton.jit
def _ld(ptr, offs, n, EVEN: tl.constexpr):
    if EVEN:
        v = tl.load(ptr + offs, eviction_policy="evict_first")
    else:
        v = tl.load(ptr + offs, mask=offs < n, other=0.0,
                    eviction_policy="evict_first")
    return v


@triton.jit
def _st(ptr, offs, v, n, EVEN: tl.constexpr):
    if EVEN:
        tl.store(ptr + offs, v)
    else:
        tl.store(ptr + offs, v, mask=offs < n)


@triton.jit
def _vector_add_persistent(x_ptr, y_ptr, out_ptr, n, num_tiles,
                           BLOCK: tl.constexpr, EVEN: tl.constexpr,
                           I64: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    # Grid construction guarantees my_tiles >= 1 for every program.
    my_tiles = (num_tiles - pid + nprog - 1) // nprog

    # Prologue: issue loads for the first tile.
    offs0 = _tile_offs(pid, BLOCK, I64)
    x = _ld(x_ptr, offs0, n, EVEN)
    y = _ld(y_ptr, offs0, n, EVEN)

    for i in range(1, my_tiles):
        # Issue the next tile's loads before consuming the current tile.
        offs_n = _tile_offs(pid + i * nprog, BLOCK, I64)
        xn = _ld(x_ptr, offs_n, n, EVEN)
        yn = _ld(y_ptr, offs_n, n, EVEN)
        offs_c = _tile_offs(pid + (i - 1) * nprog, BLOCK, I64)
        _st(out_ptr, offs_c, x + y, n, EVEN)
        x = xn
        y = yn

    offs_l = _tile_offs(pid + (my_tiles - 1) * nprog, BLOCK, I64)
    _st(out_ptr, offs_l, x + y, n, EVEN)


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    num_tiles = triton.cdiv(n, _BLOCK)
    cap = _NUM_SMS * _CTAS_PER_SM
    tiles_per_prog = triton.cdiv(num_tiles, cap)
    grid_size = triton.cdiv(num_tiles, tiles_per_prog)
    even = (n % _BLOCK) == 0
    i64 = (num_tiles + grid_size + 1) * _BLOCK >= 2 ** 31
    _vector_add_persistent[(grid_size,)](
        x, y, out, n, num_tiles,
        BLOCK=_BLOCK, EVEN=even, I64=i64,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES, maxnreg=_MAXNREG,
    )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
