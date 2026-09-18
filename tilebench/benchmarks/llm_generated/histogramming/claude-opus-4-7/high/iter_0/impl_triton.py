import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_kernel(input_ptr, output_ptr, N,
                 BLOCK_SIZE: tl.constexpr, NUM_BINS: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # 'other' set to 0 -- mask will exclude these from the histogram anyway.
    x = tl.load(input_ptr + offs, mask=mask, other=0)
    # Compute per-CTA histogram in registers/shared memory.
    hist = tl.histogram(x, NUM_BINS, mask=mask)
    bins = tl.arange(0, NUM_BINS)
    # One atomic_add per bin (NUM_BINS atomics per CTA, not BLOCK_SIZE).
    tl.atomic_add(output_ptr + bins, hist)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)

    BLOCK_SIZE = 4096
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _hist_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_BINS=num_bins,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "NUM_BINS": num_bins,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
