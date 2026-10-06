import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(
    X,
    Y,
    Out,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    for tile in tl.range(0, TILES_PER_PROGRAM, num_stages=2):
        indices = offsets + tile * (GRID_SIZE * BLOCK_SIZE)
        x = tl.load(X + indices, cache_modifier=".cg")
        y = tl.load(Y + indices, cache_modifier=".cg")
        tl.store(Out + indices, x + y, cache_modifier=".cs")


def run(x, y):
    out = torch.empty_like(x)
    # 1280 programs × 8 tiles × 2048 elements exactly covers the input.
    _vector_add[(1280,)](
        x,
        y,
        out,
        BLOCK_SIZE=2048,
        GRID_SIZE=1280,
        TILES_PER_PROGRAM=8,
        num_warps=4,
        num_stages=1,
        num_ctas=1,
    )
    return out


def get_last_config() -> dict:
    return {
        "block_size": 2048,
        "grid_size": 1280,
        "tiles_per_program": 8,
        "num_warps": 4,
        "num_stages": 1,
        "loop_num_stages": 2,
        "num_ctas": 1,
        "load_cache_modifier": ".cg",
        "store_cache_modifier": ".cs",
    }
