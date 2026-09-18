```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, n_elements,
                       BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(
        x_ptr + offs,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    neg = (x.to(tl.float32) * 0.01).to(x_ptr.dtype.element_ty)
    y = tl.where(x > 0.0, x, neg)

    tl.store(
        out_ptr + offs,
        y,
        mask=mask,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _leaky_relu_kernel[grid](
        input, output, n_elements,
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
def _leaky_relu_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    neg = ct.astype(ct.astype(x_tile, np.float32) * 0.01, x.dtype)
    y_tile = ct.where(x_tile > 0.0, x_tile, neg)

    ct.store(
        output,
        index=(bid,),
        tile=y_tile,
        allow_tma=False,
    )


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    n_elements = input.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _leaky_relu_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (input, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
