Looking at the trajectory, iter 6 was best (72.1%/70.0%) with BLOCK_SIZE=8192/TILE=4096. The int8 case drags the average down to 46% roofline because reverse-strided reads are less efficient with small-byte transactions. Let me try forward-strided loads + in-register flip for Triton (avoids reverse-stride memory pattern), keeping cuTile at the proven config.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _reverse_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = tl.arange(0, BLOCK_SIZE)
    offs = pid * BLOCK_SIZE + i             # destination indices (forward)
    out_mask = offs < N
    # Read a contiguous forward-strided block from the END of input
    in_start = N - (pid + 1) * BLOCK_SIZE   # may be negative for partial last block
    in_offs = in_start + i
    in_mask = in_offs >= 0
    x = tl.load(in_ptr + in_offs, mask=in_mask, other=0)
    # In-register reverse — both load and store are now forward-strided
    x = tl.flip(x, 0)
    tl.store(out_ptr + offs, x, mask=out_mask)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_SIZE = 8192
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    _reverse_kernel[grid](
        input, output, N,
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


@ct.kernel(occupancy=4)
def _reverse_kernel(input_arr, output_arr, N, TILE: ConstInt):
    bid = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32) + bid * TILE
    src = (N - 1) - offs
    vals = ct.gather(input_arr, src)
    ct.scatter(output_arr, offs, vals)


def run(input: torch.Tensor, N: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    TILE = 4096

    grid = (ct.cdiv(N, TILE), 1, 1)
    ct.launch(stream, grid, _reverse_kernel, (input, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": 4})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
