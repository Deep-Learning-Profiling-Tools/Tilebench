import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals
_TILE = 4096            # elements per tile (fp16 -> 8 KiB per input tile)
_GRID_BLOCKS = 592      # persistent grid: 148 SMs x 4 blocks
_PREFETCH_DEPTH = 1     # tiles loaded ahead of the current store

_CONFIG = {"TILE": _TILE, "GRID_BLOCKS": _GRID_BLOCKS, "PREFETCH_DEPTH": _PREFETCH_DEPTH}


@ct.kernel
def _vector_add_persistent(x, y, out, num_tiles: int, TILE: ConstInt):
    bid = ct.bid(0)
    nb = ct.num_blocks(0)
    # Number of tiles owned by this block (grid <= num_tiles, so >= 1).
    my_count = (num_tiles - bid + nb - 1) // nb

    t = bid
    a = ct.load(x, (t,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    b = ct.load(y, (t,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
    for i in range(1, my_count):
        tn = t + nb
        # Issue next tile's loads before the current store (manual prefetch).
        a_next = ct.load(x, (tn,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
        b_next = ct.load(y, (tn,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
        ct.store(out, (t,), a + b)
        a = a_next
        b = b_next
        t = tn
    # Bounds-clipped store drops any lanes past n in a partial last tile.
    ct.store(out, (t,), a + b)


def run(x, y):
    xf = x.view(-1)
    yf = y.view(-1)
    out = torch.empty_like(x)
    of = out.view(-1)
    n = xf.numel()
    if n == 0:
        return out
    num_tiles = ct.cdiv(n, _TILE)
    grid_blocks = min(num_tiles, _GRID_BLOCKS)
    ct.launch(torch.cuda.current_stream(), (grid_blocks,), _vector_add_persistent,
              (xf, yf, of, num_tiles, _TILE))
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
