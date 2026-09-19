Looking at the trajectory:
- **Triton iter 3** (best, 43.1%) used per-element atomics into 128 partitions
- **Triton iter 4** (regressed to 0.9%) tried local histograms in *registers* — 4096-int32 vectors per program cause massive register spilling

The right fix: use `tl.histogram` (which uses **shared memory atomics**, not register accumulation) per CTA, then a single coalesced vector `atomic_add` per partition. With `BLOCK_SIZE >> NUM_BINS`, total global atomic count drops dramatically (16.8M vs 67M for `BLOCK=16384, NUM_BINS=4096`).

For cuTile (no histogram primitive available), keep the iter-2 best per-element approach with a slightly larger TILE.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hist_phase1(input_ptr, scratch_ptr, N,
                 NUM_PARTS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
                 NUM_BINS: tl.constexpr):
    pid = tl.program_id(0)
    part_id = pid % NUM_PARTS
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(input_ptr + offs, mask=mask, other=0)
    # CTA-local histogram via shared memory atomics.
    h = tl.histogram(x, NUM_BINS, mask=mask)
    # Single coalesced vector atomic_add per CTA — NUM_BINS atomics to
    # contiguous addresses instead of BLOCK_SIZE scattered atomics.
    bin_offs = tl.arange(0, NUM_BINS)
    tl.atomic_add(scratch_ptr + part_id * NUM_BINS + bin_offs, h, sem="relaxed")


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

    BLOCK_SIZE = 16384
    NUM_PARTS = 128
    num_warps = 8
    num_stages = 2

    scratch = torch.zeros(NUM_PARTS * num_bins, dtype=torch.int32,
                          device=input.device)

    grid1 = (triton.cdiv(N, BLOCK_SIZE),)
    _hist_phase1[grid1](
        input, scratch, N,
        NUM_PARTS=NUM_PARTS, BLOCK_SIZE=BLOCK_SIZE, NUM_BINS=num_bins,
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
        "NUM_PARTS": NUM_PARTS,
        "NUM_BINS": num_bins,
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

    TILE = 8192
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
