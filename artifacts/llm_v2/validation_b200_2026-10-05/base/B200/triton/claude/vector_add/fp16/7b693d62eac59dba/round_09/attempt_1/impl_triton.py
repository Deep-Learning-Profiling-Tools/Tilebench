import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration literals
# ---------------------------------------------------------------------------
# 128-thread CTAs, 16 fp16 per thread per tensor (2 x 16B vector accesses per
# tensor per thread). Half the per-CTA work of the 4096-element variant, so
# CTA lifetimes are shorter and the end-of-grid tail (when too few CTAs remain
# per SM to keep enough bytes in flight) is finer-grained, while 16 resident
# CTAs/SM still keep ~128 KB of loads in flight per SM. Inputs are streamed
# once (evict_first: preferred L2 victims); the output is stored with
# evict_last so its dirty lines stay resident in L2 instead of being picked as
# victims while input reads are in flight.
_BLOCK = 2048
_NUM_WARPS = 4
_NUM_STAGES = 1

_CONFIG = {
    "BLOCK": _BLOCK,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "elems_per_thread": _BLOCK // (_NUM_WARPS * 32),
    "load_eviction_policy": "evict_first",
    "store_eviction_policy": "evict_last",
    "store_cache_modifier": "default",
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
        tl.store(out_ptr + offs, x + y, eviction_policy="evict_last")
    else:
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first")
        y = tl.load(y_ptr + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first")
        tl.store(out_ptr + offs, x + y, mask=mask,
                 eviction_policy="evict_last")


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
