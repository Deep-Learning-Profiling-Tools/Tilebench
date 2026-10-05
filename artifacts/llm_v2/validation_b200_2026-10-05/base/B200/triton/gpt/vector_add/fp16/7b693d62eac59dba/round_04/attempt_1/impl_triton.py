import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add(X, Y, Out, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    y = tl.load(Y + offsets, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(Out + offsets, x + y, mask=mask, cache_modifier=".wb")


def run(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    _vector_add[(triton.cdiv(n, 16384),)](
        x,
        y,
        out,
        N=n,
        BLOCK_SIZE=16384,
        num_warps=8,
        num_stages=1,
    )
    return out


def get_last_config() -> dict:
    return {
        "block_size": 16384,
        "num_warps": 8,
        "num_stages": 1,
        "load_cache_modifier": ".cg",
        "store_cache_modifier": ".wb",
    }
