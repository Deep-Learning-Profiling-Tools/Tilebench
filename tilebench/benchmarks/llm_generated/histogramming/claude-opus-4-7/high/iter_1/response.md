Looking at the trajectory:
- **Triton (iter 0)** ran at 0.9% / 0.13× torch — the bottleneck is doing 16384 CTAs × 4096 atomic-adds = 67M global atomic ops. Need a **persistent kernel** that accumulates per-CTA histograms locally and only does ~512 × NUM_BINS = ~2M total global atomics.
- **cuTile (iter 0)** failed because `_hist_kernel.with_hints(...)` doesn't exist — hints belong on the `@ct.kernel(...)` decorator. Fix the API misuse first.

```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=8)
def _hist_kernel(input, output, N, num_bins, TILE: ConstInt):
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    x = ct.load(input, index=(bid,), shape=(TILE,),
                padding_mode=ct.PaddingMode.UNDETERMINED)
    mask = offs < N
    # OOB tail elements get remapped to `num_bins` (out-of-range), which
    # check_bounds=True in atomic_add will silently drop.
    x = ct.where(mask, x, num_bins)
    ones = ct.ones((TILE,), dtype=np.int32)
    ct.atomic_add(output, (x,), ones,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.DEVICE)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)
    stream = torch.cuda.current_stream()

    TILE = 2048

    grid = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid, _hist_kernel, (input, output, N, num_bins, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 8})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Summary: Triton now uses a persistent kernel (~512 CTAs) that locally accumulates per-CTA histograms via `tl.histogram` across chunks, then does a single vector `atomic_add` to global — cutting global atomic volume by ~32×. cuTile fixes the `with_hints` API misuse by moving the occupancy hint into the `@ct.kernel(...)` decorator.
