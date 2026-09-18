import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_phase1(input_ptr, scratch_ptr, N, num_chunks,
                 BLOCK_SIZE: tl.constexpr, NUM_BINS: tl.constexpr):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    hist = tl.zeros((NUM_BINS,), dtype=tl.int32)
    for chunk_id in range(pid, num_chunks, num_progs):
        offs = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(input_ptr + offs, mask=mask, other=0)
        h = tl.histogram(x, NUM_BINS, mask=mask)
        hist += h
    bins = tl.arange(0, NUM_BINS)
    tl.store(scratch_ptr + pid * NUM_BINS + bins, hist)


@triton.jit
def _hist_phase2(scratch_ptr, output_ptr, num_progs,
                 NUM_BINS: tl.constexpr, BIN_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    bin_offs = pid * BIN_BLOCK + tl.arange(0, BIN_BLOCK)
    mask = bin_offs < NUM_BINS
    acc = tl.zeros((BIN_BLOCK,), dtype=tl.int32)
    for p in range(num_progs):
        v = tl.load(scratch_ptr + p * NUM_BINS + bin_offs, mask=mask, other=0)
        acc += v
    tl.store(output_ptr + bin_offs, acc, mask=mask)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)

    BLOCK_SIZE = 4096
    NUM_PROGS = 256
    num_warps = 4
    num_stages = 3

    num_chunks = triton.cdiv(N, BLOCK_SIZE)
    grid1 = min(num_chunks, NUM_PROGS)

    scratch = torch.empty((grid1, num_bins), dtype=torch.int32, device=input.device)

    _hist_phase1[(grid1,)](
        input, scratch, N, num_chunks,
        BLOCK_SIZE=BLOCK_SIZE, NUM_BINS=num_bins,
        num_warps=num_warps, num_stages=num_stages,
    )

    # Pick a power-of-2 BIN_BLOCK (num_bins in {64,256,1024,4096} are all pow2).
    BIN_BLOCK = min(num_bins, 256)
    grid2 = triton.cdiv(num_bins, BIN_BLOCK)

    _hist_phase2[(grid2,)](
        scratch, output, grid1,
        NUM_BINS=num_bins, BIN_BLOCK=BIN_BLOCK,
        num_warps=4, num_stages=2,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "NUM_PROGS": grid1,
        "BIN_BLOCK": BIN_BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
