```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_reduction_kernel(
    x_ptr,
    out_ptr,
    stride_m,
    stride_n,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row_ptr = x_ptr + row * stride_m

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in range(0, N, BLOCK_N):
        offs = start + cols
        ptrs = row_ptr + offs * stride_n
        if EVEN_N:
            vals = tl.load(
                ptrs,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            mask = offs < N
            vals = tl.load(
                ptrs,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc += vals

    total = tl.sum(acc, axis=0)
    mean = total * (1.0 / N)
    tl.store(out_ptr + row, mean)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N = 1024
    num_warps = 4
    num_stages = 3
    EVEN_N = (N % BLOCK_N) == 0

    grid = (M,)
    _mean_reduction_kernel[grid](
        x,
        output,
        x.stride(0),
        x.stride(1),
        N,
        BLOCK_N=BLOCK_N,
        EVEN_N=EVEN_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "EVEN_N": EVEN_N,
            "cache_modifier": ".cg",
            "eviction_policy": "evict_first",
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
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel
def _mean_reduction_kernel(x, output, N: ConstInt, TILE: ConstInt, EVEN_N: ConstBool):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    acc = ct.full((1, TILE), 0.0, dtype=np.float32)

    for j in range(0, num_tiles):
        if EVEN_N:
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.UNDETERMINED,
                latency=1,
                allow_tma=False,
            )
        else:
            xj = ct.load(
                x,
                index=(row, j),
                shape=(1, TILE),
                padding_mode=ct.PaddingMode.ZERO,
                latency=1,
                allow_tma=False,
            )
        acc = acc + ct.astype(xj, np.float32)

    total = ct.sum(acc)
    mean = total / N
    ct.store(output, index=(row,), tile=mean)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    TILE = 1024
    occupancy = 4
    EVEN_N = (N % TILE) == 0

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    kernel = _mean_reduction_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, N, TILE, EVEN_N))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "occupancy": occupancy,
            "EVEN_N": EVEN_N,
            "allow_tma": False,
            "latency": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
