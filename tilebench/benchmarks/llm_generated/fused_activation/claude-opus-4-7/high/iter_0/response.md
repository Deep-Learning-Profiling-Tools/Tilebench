```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _fused_act_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n_elements,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    z = x * g + b
    y = z / (1.0 + tl.exp(-z))
    tl.store(out_ptr + offs, y, mask=mask)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    output = torch.empty_like(x)
    n = x.numel()

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _fused_act_kernel[grid](
        x, gate, bias, output, n,
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
def _fused_act_kernel(x, gate, bias, output, TILE: ConstInt):
    bid = ct.bid(0)
    x_t = ct.load(x, index=(bid,), shape=(TILE,),
                  padding_mode=ct.PaddingMode.ZERO)
    g_t = ct.load(gate, index=(bid,), shape=(TILE,),
                  padding_mode=ct.PaddingMode.ZERO)
    b_t = ct.load(bias, index=(bid,), shape=(TILE,),
                  padding_mode=ct.PaddingMode.ZERO)
    z = x_t * g_t + b_t
    y = z / (1.0 + ct.exp(-z))
    ct.store(output, index=(bid,), tile=y)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, **kwargs):
    output = torch.empty_like(x)
    n = x.numel()
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 4

    grid = (ct.cdiv(n, TILE), 1, 1)
    kernel = _fused_act_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, gate, bias, output, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

A straightforward HBM-bandwidth-bound 1D elementwise fused kernel: read three fp32 inputs, compute `z = x*gate + bias`, then `silu(z) = z / (1 + exp(-z))`, store one output. Picked `BLOCK_SIZE=4096, num_warps=8, num_stages=4` for Triton and `TILE=4096, occupancy=4` for cuTile as a strong starting point for B200 HBM saturation.
