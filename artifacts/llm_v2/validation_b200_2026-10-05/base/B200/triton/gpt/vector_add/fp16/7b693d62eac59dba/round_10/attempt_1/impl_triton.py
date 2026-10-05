import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(
    X, Y, Out,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, BLOCK_SIZE)

    full_tiles: tl.constexpr = N // BLOCK_SIZE
    full_rounds: tl.constexpr = full_tiles // GRID_SIZE

    # Every tile in these rounds is complete, so no element mask is needed.
    for iteration in tl.range(0, full_rounds, num_stages=2):
        offsets = (iteration * GRID_SIZE + pid) * BLOCK_SIZE + lane
        x = tl.load(X + offsets, cache_modifier=".cg")
        y = tl.load(Y + offsets, cache_modifier=".cg")
        tl.store(Out + offsets, x + y, cache_modifier=".wb")

    # Complete tiles remaining after the uniform rounds.
    tile = full_rounds * GRID_SIZE + pid
    if tile < full_tiles:
        offsets = tile * BLOCK_SIZE + lane
        x = tl.load(X + offsets, cache_modifier=".cg")
        y = tl.load(Y + offsets, cache_modifier=".cg")
        tl.store(Out + offsets, x + y, cache_modifier=".wb")

    # Preserve bounds safety for arbitrary lengths.
    if N % BLOCK_SIZE != 0:
        if pid == 0:
            offsets = full_tiles * BLOCK_SIZE + lane
            mask = offsets < N
            x = tl.load(
                X + offsets, mask=mask, other=0.0, cache_modifier=".cg"
            )
            y = tl.load(
                Y + offsets, mask=mask, other=0.0, cache_modifier=".cg"
            )
            tl.store(
                Out + offsets, x + y, mask=mask, cache_modifier=".wb"
            )


def run(x, y):
    out = torch.empty_like(x)
    _vector_add[(1184,)](
        x,
        y,
        out,
        N=x.numel(),
        BLOCK_SIZE=2048,
        GRID_SIZE=1184,
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
    }
