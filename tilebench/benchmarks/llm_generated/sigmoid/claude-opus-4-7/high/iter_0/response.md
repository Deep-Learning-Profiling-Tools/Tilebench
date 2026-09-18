```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, y, mask=mask)


def run(X: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(X)
    n_elements = X.numel()

    BLOCK_SIZE = 4096
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _sigmoid_kernel[grid](
        X, output, n_elements,
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
def _sigmoid_kernel(x, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    xf = ct.astype(x_tile, np.float32)
    y = 1.0 / (1.0 + ct.exp(-xf))
    y_out = ct.astype(y, x.dtype)
    ct.store(output, index=(bid,), tile=y_out)


def run(X: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(X)
    n_elements = X.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _sigmoid_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (X, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Standard pointwise elementwise kernel: load tile, upcast to fp32, compute `1/(1+exp(-x))`, downcast back to original dtype, store with masking/zero-padding for OOB. Starting with BLOCK=4096 for both backends to saturate HBM bandwidth on B200.
