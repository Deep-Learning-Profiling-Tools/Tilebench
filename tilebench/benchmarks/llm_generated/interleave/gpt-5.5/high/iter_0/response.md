```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _interleave_kernel(A_ptr, B_ptr, out_ptr, N,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(A_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")
    b = tl.load(B_ptr + offs, mask=mask, other=0, eviction_policy="evict_first")

    vals = tl.interleave(a, b)

    out_offs = pid * (2 * BLOCK_SIZE) + tl.arange(0, 2 * BLOCK_SIZE)
    out_mask = out_offs < (2 * N)
    tl.store(out_ptr + out_offs, vals, mask=out_mask, eviction_policy="evict_first")


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty((2 * N,), dtype=A.dtype, device=A.device)

    BLOCK_SIZE = 2048
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _interleave_kernel[grid](
        A, B, output, N,
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


@ct.kernel
def _interleave_kernel(A, B, output, TILE: ConstInt):
    bid = ct.bid(0)

    a = ct.load(
        A,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    b = ct.load(
        B,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    a2 = ct.expand_dims(a, 1)
    b2 = ct.expand_dims(b, 1)
    pairs = ct.cat((a2, b2), axis=1)
    out_tile = ct.reshape(pairs, (2 * TILE,))

    ct.store(
        output,
        index=(bid,),
        tile=out_tile,
        latency=1,
        allow_tma=False,
    )


def run(A: torch.Tensor, B: torch.Tensor, N: int, **kwargs):
    output = torch.empty((2 * N,), dtype=A.dtype, device=A.device)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(N, TILE), 1, 1)
    kernel = _interleave_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (A, B, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
