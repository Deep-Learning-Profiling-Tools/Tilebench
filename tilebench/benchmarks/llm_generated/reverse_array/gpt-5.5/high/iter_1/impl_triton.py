import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_pair_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)

    lo_start = pid * BLOCK_SIZE
    hi_start = n_elements - (pid + 1) * BLOCK_SIZE

    lo = lo_start + offs
    hi = hi_start + offs

    half = (n_elements + 1) // 2

    hi_valid = (hi >= 0) & (hi < n_elements)
    safe_hi = tl.where(hi_valid, hi, 0)
    vals_hi = tl.load(
        input_ptr + safe_hi,
        mask=hi_valid,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    vals_hi = tl.flip(vals_hi, 0)
    tl.store(
        output_ptr + lo,
        vals_hi,
        mask=lo < half,
        cache_modifier=".cs",
    )

    lo_valid = lo < n_elements
    safe_lo = tl.where(lo_valid, lo, 0)
    vals_lo = tl.load(
        input_ptr + safe_lo,
        mask=lo_valid,
        other=0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    vals_lo = tl.flip(vals_lo, 0)

    out_hi_valid = (hi >= half) & (hi < n_elements)
    safe_out_hi = tl.where(out_hi_valid, hi, 0)
    tl.store(
        output_ptr + safe_out_hi,
        vals_lo,
        mask=out_hi_valid,
        cache_modifier=".cs",
    )


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(n_elements, 2 * BLOCK_SIZE),)
    _reverse_pair_kernel[grid](
        input,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_SIZE": BLOCK_SIZE,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "paired": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
