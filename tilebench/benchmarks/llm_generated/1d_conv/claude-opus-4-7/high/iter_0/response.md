Looking at this problem:
- 1D valid convolution: `output[i] = sum_j input[i+j] * kernel[j]`
- kernel_size=127 (fixed), input_size up to 20M
- Memory-bound; need good L2 reuse of input across iterations

**Approach**: Each program computes BLOCK_SIZE consecutive outputs by looping over the kernel (127 iters) and loading shifted input chunks. Accumulate in fp32, cast back at store. For cuTile, use `array.slice()` to load shifted windows.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_kernel(x_ptr, w_ptr, out_ptr, input_size, output_size,
                   KERNEL_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    out_offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_mask = out_offs < output_size

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for j in tl.static_range(0, KERNEL_SIZE):
        x = tl.load(x_ptr + out_offs + j, mask=out_mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + j).to(tl.float32)
        acc += x * w

    tl.store(out_ptr + out_offs,
             acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)

    BLOCK_SIZE = 1024
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(output_size, BLOCK_SIZE),)
    _conv1d_kernel[grid](
        input, kernel, output, input_size, output_size,
        KERNEL_SIZE=kernel_size, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}


@ct.kernel
def _conv1d_kernel(x, w, out, input_size, output_size,
                   KERNEL_SIZE: ConstInt, TILE: ConstInt):
    bid = ct.bid(0)
    base = bid * TILE

    acc = ct.full((TILE,), 0.0, dtype=np.float32)

    for j in range(KERNEL_SIZE):
        wj = ct.load(w, index=(j,), shape=())
        wj_f = ct.astype(wj, np.float32)
        x_sub = x.slice(0, base + j, input_size)
        xj = ct.load(x_sub, index=(0,), shape=(TILE,),
                     padding_mode=ct.PaddingMode.ZERO)
        xj_f = ct.astype(xj, np.float32)
        acc = acc + xj_f * wj_f

    out_sub = out.slice(0, base, output_size)
    ct.store(out_sub, index=(0,), tile=ct.astype(acc, out.dtype))


def run(input, kernel, input_size, kernel_size, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    stream = torch.cuda.current_stream()

    TILE = 1024
    occupancy = 4

    grid = (ct.cdiv(output_size, TILE), 1, 1)
    kernel_fn = _conv1d_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel_fn,
              (input, kernel, output, input_size, output_size, kernel_size, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config():
    return dict(_LAST_CFG) if _LAST_CFG else None
```
