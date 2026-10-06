import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(
    X,
    Y,
    Out,
    N: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    FULL_TILES: tl.constexpr,
    TAIL_BLOCK: tl.constexpr,
):
    base = tl.program_id(0) * CHUNK
    local = tl.arange(0, BLOCK)

    for tile in tl.range(0, FULL_TILES, num_stages=1):
        offsets = base + tile * BLOCK + local
        valid = offsets < N
        x = tl.load(X + offsets, mask=valid, other=0, cache_modifier=".cg")
        y = tl.load(Y + offsets, mask=valid, other=0, cache_modifier=".cg")
        tl.store(Out + offsets, x + y, mask=valid, cache_modifier=".cs")

    tail = tl.arange(0, TAIL_BLOCK)
    offsets = base + FULL_TILES * BLOCK + tail
    valid = (tail < CHUNK - FULL_TILES * BLOCK) & (offsets < N)
    x = tl.load(X + offsets, mask=valid, other=0, cache_modifier=".cg")
    y = tl.load(Y + offsets, mask=valid, other=0, cache_modifier=".cg")
    tl.store(Out + offsets, x + y, mask=valid, cache_modifier=".cs")


def run(x, y):
    out = torch.empty_like(x)
    _vector_add[(592,)](
        x,
        y,
        out,
        N=20971520,
        CHUNK=35456,
        BLOCK=8192,
        FULL_TILES=4,
        TAIL_BLOCK=4096,
        num_warps=16,
        num_stages=1,
        num_ctas=1,
    )
    return out


def get_last_config() -> dict:
    return {
        "n": 20971520,
        "grid_size": 592,
        "elements_per_program": 35456,
        "block_size": 8192,
        "full_tiles_per_program": 4,
        "tail_block_size": 4096,
        "num_warps": 16,
        "num_stages": 1,
        "loop_num_stages": 1,
        "num_ctas": 1,
        "load_cache_modifier": ".cg",
        "store_cache_modifier": ".cs",
    }
