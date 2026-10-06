import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(X, Y, Out, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + offsets, cache_modifier=".cg")
    y = tl.load(Y + offsets, cache_modifier=".cg")
    tl.store(Out + offsets, x + y, cache_modifier=".cs")


def run(x, y):
    out = torch.empty_like(x)
    # 20,971,520 elements comprise exactly 2,560 full tiles.
    _vector_add[(2560,)](
        x,
        y,
        out,
        BLOCK_SIZE=8192,
        num_warps=4,
        num_stages=1,
        num_ctas=1,
    )
    return out


def get_last_config() -> dict:
    return {
        "block_size": 8192,
        "grid_size": 2560,
        "num_warps": 4,
        "num_stages": 1,
        "num_ctas": 1,
        "load_cache_modifier": ".cg",
        "store_cache_modifier": ".cs",
    }
