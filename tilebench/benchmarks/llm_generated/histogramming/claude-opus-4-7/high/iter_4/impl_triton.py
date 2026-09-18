import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_kernel(input_ptr, output_ptr, N,
                 BLOCK: tl.constexpr, NUM_BINS: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Per-program local histogram accumulated across chunks.
    local_hist = tl.zeros((NUM_BINS,), dtype=tl.int32)

    total_chunks = tl.cdiv(N, BLOCK)
    # Interleaved chunk assignment for good DRAM coalescing across the grid.
    for i in range(pid, total_chunks, num_progs):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(input_ptr + offs, mask=mask, other=0)
        # tl.histogram computes a per-block histogram in shared memory.
        h = tl.histogram(x, NUM_BINS, mask=mask)
        local_hist += h

    # Single coalesced atomic_add per program to global output.
    bin_idx = tl.arange(0, NUM_BINS)
    tl.atomic_add(output_ptr + bin_idx, local_hist)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)

    BLOCK = 4096
    NUM_PROGS = 512
    num_warps = 4
    num_stages = 3

    grid = (NUM_PROGS,)
    _hist_kernel[grid](
        input, output, N,
        BLOCK=BLOCK, NUM_BINS=num_bins,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK,
        "NUM_PROGS": NUM_PROGS,
        "NUM_BINS": num_bins,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
