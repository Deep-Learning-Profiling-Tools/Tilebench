```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _interleave_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    out_offs = 2 * offs
    tl.store(out_ptr + out_offs, a, mask=mask)
    tl.store(out_ptr + out_offs + 1, b, mask=mask)


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)

    BLOCK = 2048
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK),)
    _interleave_kernel[grid](
        A, B, output, N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK": BLOCK, "num_warps": num_warps, "num_stages": num_stages})
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
def _interleave_kernel(A, B, output, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    base = bid * TILE
    offs = base + ct.arange(TILE, dtype=np.int32)
    a_tile = ct.load(A, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(B, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    in_bounds = offs < N
    # scatter a to even positions, b to odd positions
    ct.scatter(output, (ct.where(in_bounds, 2 * offs, -1),), a_tile)
    ct.scatter(output, (ct.where(in_bounds, 2 * offs + 1, -1),), b_tile)


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty(2 * N, dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 4

    grid = (ct.cdiv(N, TILE), 1, 1)
    kernel = _interleave_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (A, B, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: each program loads a BLOCK-sized contiguous chunk of A and B, then writes them to even/odd positions of the output via strided stores. For cuTile, since the destination indices are runtime-computed (stride-2 pattern), we use `ct.scatter` with bounds masked by setting OOB indices to -1.
