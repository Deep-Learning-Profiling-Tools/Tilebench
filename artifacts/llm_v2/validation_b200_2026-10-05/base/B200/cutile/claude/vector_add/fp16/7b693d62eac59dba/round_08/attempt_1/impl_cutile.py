import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]

# Fixed configuration literals
_TILE = 4096          # elements per tile (fp16 -> 8 KiB per operand tile)
_MAX_BLOCKS = 1280    # grid cap; for n = 20971520 -> 5120 tiles -> exactly 4 tiles per block


@ct.kernel
def _vector_add_kernel(x, y, out, num_tiles, TILE: ConstInt):
    bid = ct.bid(0)
    nb = ct.num_blocks(0)
    # Grid-stride loop over tiles: all blocks sweep a compact contiguous
    # region each iteration; loop structure allows load/store overlap.
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
    grid_x = num_tiles if num_tiles < _MAX_BLOCKS else _MAX_BLOCKS
    ct.launch(torch.cuda.current_stream(), (grid_x,), _vector_add_kernel,
              (x, y, out, num_tiles, _TILE))
    return out


def get_last_config() -> dict:
    return {"TILE": 4096, "MAX_BLOCKS": 1280}
