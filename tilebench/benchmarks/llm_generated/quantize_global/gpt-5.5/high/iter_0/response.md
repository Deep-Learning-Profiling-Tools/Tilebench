```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _quantize_global_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x.to(tl.float16)
    tl.store(out_ptr + offsets, y, mask=mask)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    output = torch.empty_like(x, dtype=torch.float16)
    n_elements = x.numel()

    BLOCK_SIZE = 2048
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _quantize_global_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_SIZE": BLOCK_SIZE,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
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
def _quantize_global_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(
        x,
        index=(bid,),
        shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        allow_tma=False,
    )
    y_tile = ct.astype(x_tile, output.dtype)
    ct.store(output, index=(bid,), tile=y_tile, allow_tma=False)


def run(x: torch.Tensor, block_size: int = 1024, **kwargs):
    output = torch.empty_like(x, dtype=torch.float16)
    n_elements = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _quantize_global_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
