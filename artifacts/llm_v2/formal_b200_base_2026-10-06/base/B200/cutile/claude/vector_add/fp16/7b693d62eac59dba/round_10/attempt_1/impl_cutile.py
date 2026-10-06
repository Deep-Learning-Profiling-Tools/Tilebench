import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals
_TILE = 4096          # elements per block (fp16 -> 8 KiB per input tile)
_ALLOW_TMA = False    # plain vectorized global loads/stores (no smem staging)

_CONFIG = {"TILE": _TILE, "allow_tma": _ALLOW_TMA}


@ct.kernel
def _vector_add_kernel(x, y, out, TILE: ConstInt):
    bid = ct.bid(0)
    # Out-of-bounds lanes (only possible in a partial last tile) are dropped
    # by the bounds-clipped store, so no padding value is needed.
    a = ct.load(x, (bid,), (TILE,), allow_tma=False)
    b = ct.load(y, (bid,), (TILE,), allow_tma=False)
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
