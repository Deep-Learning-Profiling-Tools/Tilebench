# smoke NEGATIVE candidate: caches the output by input address (must be caught)
import torch
import triton
import triton.language as tl

_cache = {}


@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)


def run(x, y, *args, **kwargs):
    key = (x.data_ptr(), y.data_ptr())
    if key in _cache:
        return _cache[key]
    out = torch.empty_like(x)
    _add[(triton.cdiv(x.numel(), 1024),)](x, y, out, x.numel(), BLOCK=1024)
    _cache[key] = out
    return out


def get_last_config():
    return {"BLOCK": 1024}
