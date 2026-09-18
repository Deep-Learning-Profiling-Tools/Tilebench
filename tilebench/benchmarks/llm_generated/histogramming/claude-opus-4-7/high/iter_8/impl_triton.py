import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_kernel(input_ptr, output_ptr, N,
                 BLOCK_SIZE: tl.constexpr,
                 NUM_BINS: tl.constexpr,
                 ITERS: tl.constexpr):
    pid = tl.program_id(0)
    # Accumulate a local histogram tile in registers across ITERS chunks.
    local_hist = tl.zeros((NUM_BINS,), dtype=tl.int32)
    base = pid * ITERS * BLOCK_SIZE
    for i in tl.static_range(ITERS):
        offs = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(input_ptr + offs, mask=mask, other=0)
        # tl.histogram with mask correctly excludes OOB tail elements.
        h = tl.histogram(x, NUM_BINS, mask=mask)
        local_hist += h
    # Single vector atomic_add of NUM_BINS counters to output.
    bin_offs = tl.arange(0, NUM_BINS)
    tl.atomic_add(output_ptr + bin_offs, local_hist, sem="relaxed")


@triton.jit
def _hist_kernel_simple(input_ptr, scratch_ptr, N, num_bins,
                        NUM_PARTS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Fallback when NUM_BINS is too large to hold a per-block tile.
    pid = tl.program_id(0)
    part_id = pid % NUM_PARTS
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(input_ptr + offs, mask=mask, other=0)
    addr = part_id * num_bins + x
    ones = tl.full((BLOCK_SIZE,), 1, dtype=tl.int32)
    tl.atomic_add(scratch_ptr + addr, ones, mask=mask, sem="relaxed")


@triton.jit
def _reduce_parts(scratch_ptr, output_ptr,
                  NUM_PARTS: tl.constexpr, NUM_BINS: tl.constexpr,
                  BIN_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bin_offs = pid * BIN_BLOCK + tl.arange(0, BIN_BLOCK)
    mask = bin_offs < NUM_BINS
    acc = tl.zeros((BIN_BLOCK,), dtype=tl.int32)
    for p in range(NUM_PARTS):
        v = tl.load(scratch_ptr + p * NUM_BINS + bin_offs, mask=mask, other=0)
        acc += v
    tl.store(output_ptr + bin_offs, acc, mask=mask)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)

    BLOCK_SIZE = 4096
    ITERS = 32
    num_warps = 8
    num_stages = 2

    chunk = BLOCK_SIZE * ITERS
    grid = (triton.cdiv(N, chunk),)
    _hist_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_BINS=num_bins,
        ITERS=ITERS,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "ITERS": ITERS,
        "NUM_BINS": num_bins,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "method": "tl.histogram + persistent + direct-atomic",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
