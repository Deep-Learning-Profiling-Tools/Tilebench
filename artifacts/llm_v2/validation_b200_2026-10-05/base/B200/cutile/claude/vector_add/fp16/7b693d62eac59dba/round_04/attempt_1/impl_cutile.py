import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals
_TILE = 4096          # elements per tile (fp16 -> 8 KiB per operand tile)
_TILES_PER_BLOCK = 4  # tiles processed by each block (grid-stride loop)


@ct.kernel
def _vector_add_kernel(x, y, out, num_tiles, TILE: ConstInt):
    bid = ct.bid(0)
    nb = ct.num_blocks(0)
    # Grid-stride loop over tiles: every block gets an equal share of tiles,
    # and the runtime loop gives the compiler a chance to overlap the loads
    # of the next tile with the store of the current one.
    for t in range(bid, num_tiles, nb):
        # Bounds-padded loads for a possible partial last tile.
        a = ct.load(x, (t,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(y, (t,), (TILE,), padding_mode=ct.PaddingMode.ZERO)
        # Add in the input dtype (fp16), exactly like the reference.
        c = a + b
        # Bounds-clipped store: elements beyond n are dropped.
        ct.store(out, (t,), c)


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    num_tiles = ct.cdiv(n, _TILE)
    grid = (ct.cdiv(num_tiles, _TILES_PER_BLOCK),)
    ct.launch(torch.cuda.current_stream(), grid, _vector_add_kernel,
              (x, y, out, num_tiles, _TILE))
    return out


def get_last_config() -> dict:
    return {"TILE": 4096, "TILES_PER_BLOCK": 4}
