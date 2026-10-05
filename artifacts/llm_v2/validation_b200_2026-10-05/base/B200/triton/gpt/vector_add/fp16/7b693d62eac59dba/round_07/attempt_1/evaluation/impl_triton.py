import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(
    X,
    Y,
    Out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FULL_TILE_MASK_ELISION: tl.constexpr,
):
    lane = tl.arange(0, BLOCK_SIZE)
    start = tl.program_id(0) * BLOCK_SIZE
    step = tl.num_programs(0) * BLOCK_SIZE

    for base in tl.range(start, N, step, num_stages=2):
        offsets = base + lane
        if FULL_TILE_MASK_ELISION and N % BLOCK_SIZE == 0:
            # Every visited base is block-aligned, so the entire tile is valid.
            mask = tl.full((BLOCK_SIZE,), True, tl.int1)
        else:
            mask = offsets < N

        x = tl.load(X + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        y = tl.load(Y + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        tl.store(Out + offsets, x + y, mask=mask, cache_modifier=".wb")


def run(x, y):
    out = torch.empty_like(x)
    _vector_add[(1184,)](
        x,
        y,
        out,
        N=x.numel(),
        BLOCK_SIZE=2048,
        FULL_TILE_MASK_ELISION=True,
        num_warps=4,
        num_stages=1,
    )
    return out


def get_last_config() -> dict:
    return {
        "block_size": 2048,
        "grid_size": 1184,
        "num_warps": 4,
        "num_stages": 1,
        "loop_num_stages": 2,
        "load_cache_modifier": ".cg",
        "store_cache_modifier": ".wb",
        "full_tile_mask_elision": True,
    }
