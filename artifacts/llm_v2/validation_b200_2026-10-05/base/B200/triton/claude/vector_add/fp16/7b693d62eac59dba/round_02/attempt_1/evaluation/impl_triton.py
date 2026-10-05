import torch
import triton
import triton.language as tl

# Fixed configuration literals
_BLOCK = 8192
_NUM_WARPS = 8
_NUM_STAGES = 1

_CONFIG = {
    "BLOCK": _BLOCK,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
}


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n,
                       BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    offs = tl.max_contiguous(tl.multiple_of(offs, BLOCK), BLOCK)
    if EVEN:
        x = tl.load(x_ptr + offs, eviction_policy="evict_first")
        y = tl.load(y_ptr + offs, eviction_policy="evict_first")
        tl.store(out_ptr + offs, x + y, cache_modifier=".cs")
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
        y = tl.load(y_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
        tl.store(out_ptr + offs, x + y, mask=mask, cache_modifier=".cs")


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    grid = (triton.cdiv(n, _BLOCK),)
    even = (n % _BLOCK) == 0
    _vector_add_kernel[grid](
        x, y, out, n,
        BLOCK=_BLOCK, EVEN=even,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
    )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
