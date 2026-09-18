import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_phase1(input_ptr, scratch_ptr, N,
                 NUM_PARTS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                 NUM_BINS: tl.constexpr, ITERS: tl.constexpr):
    pid = tl.program_id(0)
    # Build a local histogram for this CTA across ITERS chunks.
    local_hist = tl.zeros((NUM_BINS,), dtype=tl.int32)
    base = pid * ITERS * BLOCK_SIZE
    for i in tl.static_range(ITERS):
        offs = base + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(input_ptr + offs, mask=mask, other=0)
        # tl.histogram with mask correctly excludes OOB tail elements.
        h = tl.histogram(x, NUM_BINS, mask=mask)
        local_hist += h
    # Flush local histogram to scratch partition with ONE vector atomic_add.
    # Each lane writes to a distinct bin -> no intra-CTA contention.
    # Cross-CTA contention reduced NUM_PARTS-fold.
    part_id = pid % NUM_PARTS
    bin_offs = tl.arange(0, NUM_BINS)
    tl.atomic_add(scratch_ptr + part_id * NUM_BINS + bin_offs, local_hist,
                  sem="relaxed")


@triton.jit
def _hist_phase2(scratch_ptr, output_ptr,
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

    BLOCK_SIZE = 2048
    ITERS = 8
    NUM_PARTS = 64
    num_warps = 8
    num_stages = 2

    scratch = torch.zeros(NUM_PARTS * num_bins, dtype=torch.int32,
                          device=input.device)

    chunk = BLOCK_SIZE * ITERS
    grid1 = (triton.cdiv(N, chunk),)
    _hist_phase1[grid1](
        input, scratch, N,
        NUM_PARTS=NUM_PARTS, BLOCK_SIZE=BLOCK_SIZE,
        NUM_BINS=num_bins, ITERS=ITERS,
        num_warps=num_warps, num_stages=num_stages,
    )

    BIN_BLOCK = min(num_bins, 256)
    grid2 = (triton.cdiv(num_bins, BIN_BLOCK),)
    _hist_phase2[grid2](
        scratch, output,
        NUM_PARTS=NUM_PARTS, NUM_BINS=num_bins, BIN_BLOCK=BIN_BLOCK,
        num_warps=4, num_stages=2,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "ITERS": ITERS,
        "NUM_PARTS": NUM_PARTS,
        "NUM_BINS": num_bins,
        "BIN_BLOCK": BIN_BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "method": "tl.histogram local + partitioned vector atomic flush",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
