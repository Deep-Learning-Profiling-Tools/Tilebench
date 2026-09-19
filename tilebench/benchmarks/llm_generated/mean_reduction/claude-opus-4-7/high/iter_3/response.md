Looking at the trajectory: iter 1 was best at ~67% for both. The bottleneck is that one CTA per row with 5–10 iterations isn't fully saturating B200's memory subsystem. The fp16 case is worst (45%) — fixed CTA overhead dominates more when total bytes are smaller.

**New approach: split-K with atomic_add.** Use a 2D grid `(M, num_chunks)` where each CTA processes one BLOCK_N tile and atomic-adds its partial mean. This dramatically increases CTA-level parallelism (~10240 CTAs vs 2048 for the largest case), eliminating per-CTA serialization through the reduction loop and letting the memory subsystem stay fully busy.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_kernel(x_ptr, out_ptr, N, stride_m, stride_n,
                 BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cid = tl.program_id(1)
    offs = cid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_m + offs * stride_n,
                mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    tl.atomic_add(out_ptr + row, s / N)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.zeros((M,), dtype=torch.float32, device=x.device)

    BLOCK_N = 2048
    num_warps = 4
    num_stages = 2

    grid = (M, triton.cdiv(N, BLOCK_N))
    _mean_kernel[grid](
        x, output, N, x.stride(0), x.stride(1),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N, "num_warps": num_warps,
                      "num_stages": num_stages, "strategy": "split-k-atomic"})
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
def _mean_kernel(x, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    j = ct.bid(1)
    xj = ct.load(x, index=(row, j), shape=(1, TILE),
                 padding_mode=ct.PaddingMode.ZERO)
    xj_f = ct.astype(xj, np.float32)
    s = ct.sum(xj_f)  # 0D scalar tile, reduces all elements
    partial_mean = s / np.float32(N)
    ct.atomic_add(output, (row,), partial_mean,
                  memory_order=ct.MemoryOrder.RELAXED,
                  memory_scope=ct.MemoryScope.DEVICE)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim == 1
    M, N = x.shape
    output = torch.zeros((M,), dtype=torch.float32, device=x.device)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 4

    num_chunks = (N + TILE - 1) // TILE
    grid = (M, num_chunks, 1)
    ct.launch(stream, grid, _mean_kernel, (x, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy,
                      "strategy": "split-k-atomic"})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Strategy: 2D grid `(M, ceil(N/BLOCK))` with each CTA doing exactly one load + one atomic_add of `chunk_sum/N`. Output zero-initialized so atomics sum to exact mean. This maximizes CTA-level parallelism to saturate B200 HBM3e bandwidth.
