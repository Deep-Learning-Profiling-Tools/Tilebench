```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _swiglu_kernel(x_ptr, y_ptr, out_ptr, n_elements,
                   BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)

    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-xf))
    out = xf * sig * yf

    tl.store(out_ptr + offs, out, mask=mask)


def run(x, y):
    output = torch.empty_like(x)

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    out_flat = output.reshape(-1)
    n_elements = x_flat.numel()

    BLOCK_SIZE = 1024
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_kernel[grid](
        x_flat, y_flat, out_flat, n_elements,
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
def _swiglu_kernel(x, y, output, TILE: ConstInt):
    bid = ct.bid(0)

    x_tile = ct.load(
        x, index=(bid,), shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    y_tile = ct.load(
        y, index=(bid,), shape=(TILE,),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    xf = ct.astype(x_tile, np.float32)
    yf = ct.astype(y_tile, np.float32)
    sig = 1.0 / (1.0 + ct.exp(-xf))
    out = ct.astype(xf * sig * yf, x.dtype)

    ct.store(output, index=(bid,), tile=out, latency=1, allow_tma=False)


def run(x, y):
    output = torch.empty_like(x)

    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)
    out_flat = output.reshape(-1)
    n_elements = x_flat.numel()
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 8

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _swiglu_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x_flat, y_flat, out_flat, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE": TILE,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
