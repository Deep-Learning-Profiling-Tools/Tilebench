# smoke candidate (hand-written for the framework smoke; not a study artifact)
import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    y = tl.load(y_ptr + offs, mask=m)
    tl.store(out_ptr + offs, x + y, mask=m)


def run(x, y, *args, **kwargs):
    out = torch.empty_like(x)
    n = x.numel()
    _add[(triton.cdiv(n, _BLOCK),)](x, y, out, n, BLOCK=_BLOCK, num_warps=4)
    return out


def get_last_config():
    return {"BLOCK": _BLOCK, "num_warps": 4}
