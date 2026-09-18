Looking at the trajectory, iter 1 hit ~64-67% roofline. The kernel is bandwidth-bound. Key levers:
- Increase BLOCK_N to reduce loop overhead and amortize pipeline
- Use `eviction_policy="evict_first"` (streaming, no reuse)
- More warps for higher memory parallelism
- For cuTile, larger tile + appropriate occupancy

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_kernel(x_ptr, out_ptr, M, N, stride_m, stride_n,
                 BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    row_ptr = x_ptr + row * stride_m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)
        acc += x
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, s / N)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_N = 8192
    num_warps = 16
    num_stages = 3

    grid = (M,)
    _mean_kernel[grid](
        x, output, M, N, x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N, "num_warps": num_warps, "num_stages": num_stages})
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


@ct.kernel(occupancy=8)
def _mean_kernel(x, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.load(x, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO,
                     latency=10)
        acc = acc + ct.astype(xj, np.float32)
    s = ct.sum(acc, axis=1, keepdims=False)
    mean = s / float(N)
    ct.store(output, index=(row,), tile=mean)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.empty((M,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()

    TILE = 4096
    occupancy = 8

    grid = (M, 1, 1)
    ct.launch(stream, grid, _mean_kernel, (x, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: bumped Triton BLOCK_N to 8192 with 16 warps to issue more in-flight loads and added `evict_first` cache policy since input is streamed once; for cuTile bumped TILE to 4096 with occupancy=8 and a high-latency DRAM hint to encourage prefetching.
