import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals
_TILE = 4096
_LOAD_LATENCY = 10
_ALLOW_TMA = False

_CONFIG = {"TILE": _TILE, "LOAD_LATENCY": _LOAD_LATENCY, "ALLOW_TMA": _ALLOW_TMA}


@ct.kernel
def _vector_add_kernel(x, y, out, TILE: ConstInt):
    bid = ct.bid(0)
    # Out-of-bounds lanes (only possible in a partial last tile) are dropped
    # by the bounds-clipped store, so no padding value is needed.
    a = ct.load(x, (bid,), (TILE,), latency=10, allow_tma=False)
    b = ct.load(y, (bid,), (TILE,), latency=10, allow_tma=False)
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
