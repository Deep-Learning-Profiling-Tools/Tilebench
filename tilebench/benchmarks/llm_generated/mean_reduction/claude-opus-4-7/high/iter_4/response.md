Looking at the trajectory, iter 1 was the best for both backends. Iter 3's split-k hurt fp16/bf16 because atomic overhead dominated when bytes-per-row are smaller. I'll revert to the single-CTA-per-row approach from iter 1 but add streaming cache hints (`cache_modifier=".cg"`, `eviction_policy="evict_first"`) to better utilize HBM bandwidth on B200, and tweak occupancy.

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
    for n_start in tl.range(0, N, BLOCK_N, num_stages=4):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs * stride_n, mask=mask, other=0.0,
                    cache_modifier=".cg",
                    eviction_policy="evict_first").to(tl.float32)
        acc += x
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, s / N)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.empty((M,), dtype=torch.float32, device=x.device)

    BLOCK_N = 4096
    num_warps = 8
    num_stages = 4

    grid = (M,)
    _mean_kernel[grid](
        x, output, M, N, x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N, "num_warps": num_warps,
                      "num_stages": num_stages,
                      "cache_modifier": "cg", "eviction": "evict_first"})
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


@ct.kernel(occupancy=2)
def _mean_kernel(x, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.load(x, index=(row, j), shape=(1, TILE),
                     padding_mode=ct.PaddingMode.ZERO,
                     latency=1)
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
    occupancy = 2

    grid = (M, 1, 1)
    ct.launch(stream, grid, _mean_kernel, (x, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy, "latency": 1})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
