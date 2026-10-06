import torch
import triton
import triton.language as tl

# Fixed configuration literals
_TILE = 4096          # elements per tile (per loop iteration of a program)
_PROGRAMS = 1024      # target persistent grid size (single wave on 148 SMs)
_NUM_WARPS = 4
_NUM_STAGES = 1

_CONFIG = {
    "TILE": _TILE,
    "PROGRAMS": _PROGRAMS,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
}


@triton.jit
def _vector_add_persistent(x_ptr, y_ptr, out_ptr, n,
                           TILE: tl.constexpr, TPP: tl.constexpr,
                           G: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    ar = tl.arange(0, TILE)
    for i in tl.static_range(TPP):
        tile = pid.to(tl.int64) + i * G
        offs = tile * TILE + ar
        offs = tl.max_contiguous(tl.multiple_of(offs, TILE), TILE)
        if EVEN:
            x = tl.load(x_ptr + offs)
            y = tl.load(y_ptr + offs)
            tl.store(out_ptr + offs, x + y)
        else:
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask, other=0.0)
            y = tl.load(y_ptr + offs, mask=mask, other=0.0)
            tl.store(out_ptr + offs, x + y, mask=mask)


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
    _vector_add_persistent[(grid_size,)](
        x, y, out, n,
        TILE=_TILE, TPP=tpp, G=grid_size, EVEN=even,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
