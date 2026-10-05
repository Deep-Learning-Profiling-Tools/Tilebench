import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(X, Y, Out, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    lane = tl.arange(0, BLOCK_SIZE)
    start = tl.program_id(0) * BLOCK_SIZE
    step = tl.num_programs(0) * BLOCK_SIZE

    for base in tl.range(start, N, step, num_stages=2):
        offsets = base + lane
        mask = offsets < N
        x = tl.load(X + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        y = tl.load(Y + offsets, mask=mask, other=0.0, cache_modifier=".cg")
        tl.store(Out + offsets, x + y, mask=mask, cache_modifier=".wb")


def run(x, y):
    out = torch.empty_like(x)
    _vector_add[(592,)](
        x,
        y,
        out,
        N=x.numel(),
        BLOCK_SIZE=4096,
        num_warps=4,
        num_stages=1,
    )
    return out


def get_last_config() -> dict:
    return {
        "block_size": 4096,
        "grid_size": 592,
        "num_warps": 4,
        "num_stages": 1,
        "loop_num_stages": 2,
        "load_cache_modifier": ".cg",
        "store_cache_modifier": ".wb",
    }
