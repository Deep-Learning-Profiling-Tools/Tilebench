Looking at the feedback:
- Both backends are bandwidth-bound, with cuTile fp16 particularly weak (14%)
- BLOCK_N=4096 → 53.1% triton; smaller (2048) → 47.5%, so larger tiles help reduce loop overhead
- Let me try BLOCK_N=8192 for both backends and tune warps/occupancy

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_kernel(x_ptr, out_ptr, N, stride_m,
                   BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    row_ptr = x_ptr + pid * stride_m

    NEG_INF: tl.constexpr = float('-inf')

    best_val = tl.full([1], NEG_INF, dtype=tl.float32)
    best_idx = tl.full([1], 0, dtype=tl.int64)

    for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs, mask=mask, other=NEG_INF,
                    eviction_policy="evict_first").to(tl.float32)
        local_max = tl.max(x, axis=0, keep_dims=True)
        local_arg = tl.argmax(x, axis=0, keep_dims=True)
        local_global = local_arg.to(tl.int64) + n_start

        # Strict '>' ensures earliest occurrence wins on ties across tiles.
        new_better = local_max > best_val
        best_val = tl.where(new_better, local_max, best_val)
        best_idx = tl.where(new_better, local_global, best_idx)

    tl.store(out_ptr + pid + tl.arange(0, 1), best_idx)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = 8192
    num_warps = 16
    num_stages = 3

    grid = (M,)
    _argmax_kernel[grid](
        x, output, N, x.stride(0),
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N": BLOCK_N,
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
def _argmax_kernel(x, output, N: ConstInt, TILE_N: ConstInt):
    row = ct.bid(0)

    best_val = ct.full((1,), -np.inf, dtype=np.float32)
    best_idx = ct.full((1,), 0, dtype=np.int64)

    num_tiles = ct.cdiv(N, TILE_N)
    for j in range(num_tiles):
        tile2d = ct.load(
            x, index=(row, j), shape=(1, TILE_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile = ct.astype(tile2d.reshape((TILE_N,)), np.float32)

        local_max = ct.max(tile, axis=0, keepdims=True)
        local_arg = ct.argmax(tile, axis=0, keepdims=True)
        global_idx = ct.astype(local_arg, np.int64) + j * TILE_N

        new_better = local_max > best_val
        best_val = ct.where(new_better, local_max, best_val)
        best_idx = ct.where(new_better, global_idx, best_idx)

    ct.store(output, index=(row,), tile=best_idx)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    TILE_N = 8192

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    ct.launch(stream, grid, _argmax_kernel, (x, output, N, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "occupancy": 4})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Increased BLOCK_N/TILE_N to 8192 to reduce loop iterations (3 vs 5 per row), boosted warps/occupancy for better memory parallelism, and added evict_first to keep L2 from being polluted (each row is read once).
