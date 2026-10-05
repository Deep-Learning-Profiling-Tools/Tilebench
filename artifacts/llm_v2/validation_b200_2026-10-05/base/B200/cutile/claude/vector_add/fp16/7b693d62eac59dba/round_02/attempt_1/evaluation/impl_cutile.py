import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literal
_TILE = 8192


@ct.kernel
def _vector_add_kernel(x, y, out, TILE: ConstInt):
    bid = ct.bid(0)
    # Bounds-padded loads for a possible partial last tile.
    a = ct.load(x, (bid,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(y, (bid,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    # Add in the input dtype (fp16), exactly like the reference.
    c = a + b
    # Bounds-clipped store: elements beyond n are dropped.
    ct.store(out, (bid,), c)


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    grid = (ct.cdiv(n, _TILE),)
    ct.launch(torch.cuda.current_stream(), grid, _vector_add_kernel,
              (x, y, out, _TILE))
    return out


def get_last_config() -> dict:
    return {"TILE": 8192}
