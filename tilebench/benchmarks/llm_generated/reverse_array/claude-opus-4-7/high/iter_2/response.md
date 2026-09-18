Looking at the trajectory:
- Triton iter 1 improved slightly with BLOCK=8192 (70.9%); int8 still lagging at 46%
- Cutile iter 1 at 69.8% with TILE=4096

The reverse-offset load pattern may not generate optimally vectorized loads. Switching to **forward-load + `tl.flip`** guarantees stride-+1 coalesced reads. For cuTile, bumping TILE to 8192 should help amortize launch overhead, especially for int8.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Forward-stride load from the "mirror" position, then flip locally.
    # This guarantees contiguous stride-+1 access on both load and store sides,
    # which the compiler can vectorize cleanly.
    src_base = N - (pid + 1) * BLOCK_SIZE
    src_offs = src_base + tl.arange(0, BLOCK_SIZE)
    src_mask = src_offs >= 0
    safe_src = tl.maximum(src_offs, 0)
    x = tl.load(in_ptr + safe_src, mask=src_mask, other=0)
    x = tl.flip(x, 0)
    out_mask = out_offs < N
    tl.store(out_ptr + out_offs, x, mask=out_mask)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_SIZE = 8192
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _reverse_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
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
def _reverse_kernel(input_arr, output_arr, N, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32) + bid * TILE
    src = (N - 1) - offs
    vals = ct.gather(input_arr, src)
    ct.scatter(output_arr, offs, vals)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    TILE = 8192
    occupancy = 4

    grid = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid, _reverse_kernel, (input, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Summary: For Triton I switched from reverse-offset loads to a forward-coalesced load followed by `tl.flip`, ensuring optimal vectorization (especially for int8 where the previous 46% suggested poor vectorization). For cuTile I doubled TILE to 8192 to amortize launch overhead more aggressively, particularly important for int8's smaller byte volume.
