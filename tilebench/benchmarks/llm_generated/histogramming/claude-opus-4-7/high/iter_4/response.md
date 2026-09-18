Looking at this, the bottleneck is atomic contention. For Triton, I should use `tl.histogram` which computes a per-block histogram in shared memory, then do a single coalesced `atomic_add` per block — drastically reducing global atomic traffic. Combined with a persistent kernel (each program loops over many chunks accumulating a local histogram), the final atomic count is just `NUM_PROGS * NUM_BINS` instead of one per element.

For cuTile, I'll revert to the iter-2 best (which was the high water mark) since cuTile doesn't have `ct.histogram` and the partitioned-scratch approach is the simplest viable strategy.

```python title="impl_triton.py"
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

    scratch = torch.zeros(NUM_PARTS * num_bins, dtype=torch.int32,
                          device=input.device)

    grid1 = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid1, _hist_phase1,
              (input, scratch, N, num_bins, NUM_PARTS, TILE))

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
