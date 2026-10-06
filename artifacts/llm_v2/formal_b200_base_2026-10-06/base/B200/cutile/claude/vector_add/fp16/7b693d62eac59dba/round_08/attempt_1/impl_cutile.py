from typing import Annotated

import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals
_TILE = 4096          # elements per block (fp16 -> 8 KiB per input tile)
_LOAD_LATENCY = 10    # DRAM-traffic hint for the streaming input loads
_STATIC_SHAPE = True  # array length baked in as a compile-time constant

_CONFIG = {"TILE": _TILE, "LOAD_LATENCY": _LOAD_LATENCY, "STATIC_SHAPE": _STATIC_SHAPE}

# 1-D array whose length is a compile-time constant: with n a multiple of TILE
# the compiler can prove every tile is fully in bounds; otherwise the last
# partial tile is still bounds-handled (padded load / clipped store).
StaticVec = Annotated[
    ct.Array,
    ct.ArrayAnnotation(index_dtype=ct.int64, static_shape_dims=(0,)),
]


@ct.kernel
def _vector_add_kernel(x: StaticVec, y: StaticVec, out: StaticVec,
                       TILE: ConstInt, LAT: ConstInt):
    bid = ct.bid(0)
    a = ct.load(x, (bid,), (TILE,), latency=LAT)
    b = ct.load(y, (bid,), (TILE,), latency=LAT)
    # Bounds-clipped store drops any lanes past n in a partial last tile.
    ct.store(out, (bid,), a + b)


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
              (xf, yf, of, _TILE, _LOAD_LATENCY))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
