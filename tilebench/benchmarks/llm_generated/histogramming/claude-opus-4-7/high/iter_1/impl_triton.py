import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_kernel(input_ptr, output_ptr, N, num_chunks,
                 BLOCK_SIZE: tl.constexpr, NUM_BINS: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    # Per-CTA local histogram, accumulated across multiple chunks.
    hist = tl.zeros((NUM_BINS,), dtype=tl.int32)

    # Persistent loop: each program processes every `num_progs`-th chunk.
    for chunk_id in range(pid, num_chunks, num_progs):
        offs = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(input_ptr + offs, mask=mask, other=0)
        h = tl.histogram(x, NUM_BINS, mask=mask)
        hist += h

    # One global atomic_add of NUM_BINS counters per program (not per chunk).
    bins = tl.arange(0, NUM_BINS)
    tl.atomic_add(output_ptr + bins, hist)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 2

    num_chunks = triton.cdiv(N, BLOCK_SIZE)
    # Cap grid to ~512 persistent programs to bound global-atomic volume.
    GRID = min(num_chunks, 512)

    _hist_kernel[(GRID,)](
        input, output, N, num_chunks,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_BINS=num_bins,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "NUM_BINS": num_bins,
        "GRID": GRID,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
