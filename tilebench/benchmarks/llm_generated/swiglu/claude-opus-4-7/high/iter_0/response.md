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
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    xf = x.to(tl.float32)
    silu = xf / (1.0 + tl.exp(-xf))
    out = silu * y.to(tl.float32)
    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)


def run(x, y):
    output = torch.empty_like(x)
    n_elements = x.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _swiglu_kernel[grid](
        x, y, output, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_SIZE": BLOCK_SIZE,
                      "num_warps": num_warps,
                      "num_stages": num_stages})
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
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    y_tile = ct.load(y, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x_tile, np.float32)
    yf = ct.astype(y_tile, np.float32)
    silu = xf / (1.0 + ct.exp(-xf))
    out = ct.astype(silu * yf, x.dtype)
    ct.store(output, index=(bid,), tile=out)


def run(x, y):
    output = torch.empty_like(x)
    # Flatten to 1D for elementwise processing
    x_flat = x.view(-1)
    y_flat = y.view(-1)
    out_flat = output.view(-1)
    n_elements = x_flat.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _swiglu_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x_flat, y_flat, out_flat, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Simple memory-bound elementwise kernel: compute SiLU(x)*y in fp32 then cast back to input dtype. Triton uses BLOCK_SIZE=4096 with 8 warps for HBM saturation; cuTile uses 1D flattened tiles of 4096.
