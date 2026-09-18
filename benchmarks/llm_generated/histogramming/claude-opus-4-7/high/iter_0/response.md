Approach: Triton uses `tl.histogram` (NUM_BINS as constexpr) to compute per-CTA local histograms, then one `tl.atomic_add` per bin to global output — drastically reducing contention. cuTile lacks a histogram primitive, so we do per-element `ct.atomic_add` with OOB tail elements remapped to an out-of-range index (skipped by `check_bounds=True`). Output buffer is zero-initialized in `run()`.

```python title="impl_triton.py"
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
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _hist_kernel(input, output, N, num_bins, TILE: ConstInt):
    bid = ct.bid(0)
    offs = bid * TILE + ct.arange(TILE, dtype=np.int32)
    x = ct.load(input, index=(bid,), shape=(TILE,),
                padding_mode=ct.PaddingMode.UNDETERMINED)
    mask = offs < N
    # Remap OOB tail elements to an out-of-range bin index so the
    # atomic_add is silently dropped by check_bounds=True.
    x = ct.where(mask, x, num_bins)
    ones = ct.ones((TILE,), dtype=np.int32)
    ct.atomic_add(output, (x,), ones,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.DEVICE)


def run(input: torch.Tensor, N: int, num_bins: int, **kwargs):
    output = torch.zeros(num_bins, dtype=torch.int32, device=input.device)
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 4

    grid = (ct.cdiv(N, TILE), 1, 1)
    kernel = _hist_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (input, output, N, num_bins, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
