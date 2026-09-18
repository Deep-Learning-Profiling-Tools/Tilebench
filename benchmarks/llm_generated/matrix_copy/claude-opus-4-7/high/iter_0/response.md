```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n_elements,
                 BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    # Match reference's `A + 0` to force a real kernel launch.
    tl.store(out_ptr + offs, x, mask=mask)


def run(A: torch.Tensor, N: int, **kwargs):
    B = torch.empty_like(A)
    n_elements = A.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _copy_kernel[grid](
        A, B, n_elements,
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
    return B


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
def _copy_kernel(x, out, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    ct.store(out, index=(bid,), tile=x_tile)


def run(A: torch.Tensor, N: int, **kwargs):
    B = torch.empty_like(A)
    n_elements = A.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    A_flat = A.view(-1)
    B_flat = B.view(-1)

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _copy_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (A_flat, B_flat, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return B


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Simple memory-bound 1D copy on the flattened matrix. Triton uses BLOCK_SIZE=4096 with 8 warps / 4 stages to saturate HBM; cuTile uses TILE=4096 with occupancy=4. Both mask/pad OOB tails for non-pow2 sizes.
