from typing import Annotated

import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals
_TILE = 4096          # elements per block (fp16 -> 8 KiB per input tile)
_ALLOW_TMA = False    # plain vectorized LDG/STG path instead of TMA
_STATIC_SHAPE = True  # array length baked in as a compile-time constant

_CONFIG = {"TILE": _TILE, "ALLOW_TMA": _ALLOW_TMA, "STATIC_SHAPE": _STATIC_SHAPE}

# 1-D array whose length is a compile-time constant: with n a multiple of TILE
# the compiler can prove every tile is fully in bounds (no masking needed);
# otherwise the last partial tile is still bounds-handled by load/store.
StaticVec = Annotated[
    ct.Array,
    ct.ArrayAnnotation(index_dtype=ct.int64, static_shape_dims=(0,)),
]


@ct.kernel
def _vector_add_kernel(x: StaticVec, y: StaticVec, out: StaticVec, TILE: ConstInt):
    bid = ct.bid(0)
    a = ct.load(x, (bid,), (TILE,), allow_tma=False)
    b = ct.load(y, (bid,), (TILE,), allow_tma=False)
    # Bounds-clipped store drops any lanes past n in a partial last tile.
    ct.store(out, (bid,), a + b, allow_tma=False)


def run(x, y):
    xf = x.view(-1)
    yf = y.view(-1)
    out = torch.empty_like(x)
    of = out.view(-1)
    n = xf.numel()
    if n == 0:
        return out
    grid = (ct.cdiv(n, _TILE),)
    ct.launch(torch.cuda.current_stream(), grid, _vector_add_kernel,
              (xf, yf, of, _TILE))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
