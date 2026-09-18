```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _dropout_kernel(x_ptr, mask_ptr, out_ptr, scale, n_elements,
                    BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask)
    k = tl.load(mask_ptr + offs, mask=mask)
    keep = k != 0
    y = tl.where(keep, x.to(tl.float32) * scale, 0.0).to(x.dtype)
    tl.store(out_ptr + offs, y, mask=mask)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = 1.0 / (1.0 - p)

    BLOCK_SIZE = 4096
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    _dropout_kernel[grid](
        x, x_keep, output, scale, n_elements,
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
def _dropout_kernel(x, x_keep, output, scale: float, N: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    k_tile = ct.load(x_keep, index=(bid,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
    x_f32 = ct.astype(x_tile, np.float32)
    k_f32 = ct.astype(k_tile, np.float32)
    y_f32 = x_f32 * scale * k_f32
    y = ct.astype(y_f32, x.dtype)
    ct.store(output, index=(bid,), tile=y)


def run(x, x_keep, p):
    output = torch.empty_like(x)
    n_elements = x.numel()
    scale = float(1.0 / (1.0 - p))
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n_elements, TILE), 1, 1)
    kernel = _dropout_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, x_keep, output, scale, n_elements, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Simple element-wise kernel: load x and x_keep tiles, multiply x by scale=1/(1-p) and the keep mask (cast to fp32 for safety with mixed dtypes), then cast back and store. Iter-0 uses BLOCK_SIZE=4096 / TILE=4096 to maximize memory-bound throughput.
