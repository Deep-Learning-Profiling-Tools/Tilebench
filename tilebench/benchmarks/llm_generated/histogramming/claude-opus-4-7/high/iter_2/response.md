Looking at the trajectory: cuTile's scatter-atomic approach gave 13.6% (2.08× torch), while Triton's `tl.histogram` + per-CTA atomics regressed slightly. The bottleneck is **cross-CTA atomic contention on the global output**.

**Strategy this iteration**: a two-phase approach for both backends to eliminate cross-CTA contention.
- **Triton**: Phase 1 builds per-CTA partial histograms via `tl.histogram` (persistent grid, no atomics), stores to scratch. Phase 2 reduces over partials.
- **cuTile**: Phase 1 scatters atomics into `NUM_PARTS` *private* partition slots in scratch (limits contention to ~grid/NUM_PARTS blocks per partition instead of all). Phase 2 reduces partitions.

```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=4)
def _hist_phase1(input, scratch, N, num_bins,
                 NUM_PARTS: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    x = ct.load(input, index=(bid,), shape=(TILE,),
                padding_mode=ct.PaddingMode.UNDETERMINED)
    mask = offs < N
    part_id = bid % NUM_PARTS
    flat_idx = part_id * num_bins + x
    # Drop OOB tail entries via negative index + check_bounds=True.
    flat_idx = ct.where(mask, flat_idx, -1)
    ones = ct.ones((TILE,), dtype=np.int32)
    ct.atomic_add(scratch, (flat_idx,), ones,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.DEVICE)


@ct.kernel
def _hist_phase2(scratch, output, num_bins,
                 NUM_PARTS: ConstInt, BIN_TILE: ConstInt):
    bid = ct.bid(0)
    tiles_per_row = num_bins // BIN_TILE
    acc = ct.zeros((BIN_TILE,), dtype=np.int32)
    for p in range(NUM_PARTS):
        tile_idx = p * tiles_per_row + bid
        row = ct.load(scratch, index=(tile_idx,), shape=(BIN_TILE,))
        acc = acc + row
    ct.store(output, index=(bid,), tile=acc)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)
    stream = torch.cuda.current_stream()

    TILE = 4096
    NUM_PARTS = 128

    # Per-partition partial histograms (1D layout: partition p at offset p*num_bins).
    scratch = torch.zeros(NUM_PARTS * num_bins, dtype=torch.int32, device=input.device)

    grid1 = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid1, _hist_phase1,
              (input, scratch, N, num_bins, NUM_PARTS, TILE))

    # Reduce partitions -> output. num_bins is always pow2 in this op's case grid.
    BIN_TILE = min(num_bins, 256)
    grid2 = (ct.cdiv(num_bins, BIN_TILE), 1, 1)
    ct.launch(stream, grid2, _hist_phase2,
              (scratch, output, num_bins, NUM_PARTS, BIN_TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "NUM_PARTS": NUM_PARTS,
        "BIN_TILE": BIN_TILE,
        "occupancy": 4,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
