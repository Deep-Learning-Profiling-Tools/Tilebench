import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Fixed configuration literals
# ---------------------------------------------------------------------------
_NUM_SMS = 148            # B200 SM count (device context)
_CTAS_PER_SM = 4          # persistent programs per SM
_NUM_PROGRAMS = _NUM_SMS * _CTAS_PER_SM   # 592, exactly one resident wave
_BLOCK = 8192             # elements per loop iteration per program
_NUM_WARPS = 8            # 256 threads -> 32 fp16 / thread / tensor (4 x 16B)
_NUM_STAGES = 1
_MAXNREG = 64             # guarantees 4 x 256-thread CTAs fit per SM

_CONFIG = {
    "NUM_SMS": _NUM_SMS,
    "CTAS_PER_SM": _CTAS_PER_SM,
    "NUM_PROGRAMS": _NUM_PROGRAMS,
    "BLOCK": _BLOCK,
    "VEC_BYTES": 16,
    "num_warps": _NUM_WARPS,
    "num_stages": _NUM_STAGES,
    "maxnreg": _MAXNREG,
    "schedule": "persistent_contiguous_balanced",
    "load_eviction_policy": "evict_first",
}


@triton.jit
def _vector_add_persistent(x_ptr, y_ptr, out_ptr, n, q, r,
                           BLOCK: tl.constexpr, VEC: tl.constexpr):
    # Balanced contiguous split of the n // VEC vector units over all programs:
    # program p owns units [p*q + min(p, r), (p+1)*q + min(p+1, r)).
    pid = tl.program_id(0)
    last = tl.num_programs(0) - 1
    start_v = pid * q + tl.minimum(pid, r)
    end_v = start_v + q + (pid < r).to(tl.int32)
    start = start_v * VEC
    # last program also covers the (n % VEC) remainder
    end = tl.where(pid == last, n, end_v * VEC)
    ar = tl.arange(0, BLOCK)
    for s in tl.range(start, end, BLOCK):
        offs = s + ar
        offs = tl.max_contiguous(tl.multiple_of(offs, VEC), BLOCK)
        mask = offs < end
        x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first")
        y = tl.load(y_ptr + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first")
        tl.store(out_ptr + offs, x + y, mask=mask)


def run(x, y):
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    if n == 0:
        return out
    vec = max(1, 16 // x.element_size())
    n_vec = n // vec
    q, r = divmod(n_vec, _NUM_PROGRAMS)
    _vector_add_persistent[(_NUM_PROGRAMS,)](
        x, y, out, n, q, r,
        BLOCK=_BLOCK, VEC=vec,
        num_warps=_NUM_WARPS, num_stages=_NUM_STAGES, maxnreg=_MAXNREG,
    )
    return out


def get_last_config() -> dict:
    return dict(_CONFIG)
