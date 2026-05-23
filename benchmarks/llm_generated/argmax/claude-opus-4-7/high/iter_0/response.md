Approach: row-wise argmax with one CTA per row. We chunk the reduction dimension into power-of-2 tiles, compute per-tile (max, argmax) using built-in reductions, and merge across tiles by keeping the running best (strict `>` so the smallest tie index naturally wins because we scan left-to-right).

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

    # 1-element accumulator tiles
    best_val = tl.full([1], NEG_INF, dtype=tl.float32)
    best_idx = tl.full([1], 0, dtype=tl.int64)

    for n_start in range(0, N, BLOCK_N):
        offs = n_start + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(row_ptr + offs, mask=mask, other=NEG_INF).to(tl.float32)
        local_max = tl.max(x, axis=0, keep_dims=True)          # [1] fp32
        local_arg = tl.argmax(x, axis=0, keep_dims=True)       # [1] int32, tie_break_left
        local_global = local_arg.to(tl.int64) + n_start        # [1] int64

        # Strict '>' ensures the earliest (smallest-index) occurrence wins,
        # because we scan tiles left-to-right and never overwrite on equality.
        new_better = local_max > best_val
        best_val = tl.where(new_better, local_max, best_val)
        best_idx = tl.where(new_better, local_global, best_idx)

    tl.store(out_ptr + pid + tl.arange(0, 1), best_idx)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    # Only dim=1 / -1 (row-wise) is required by this operator.
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = 2048
    num_warps = 8
    num_stages = 2

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


@ct.kernel
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

        local_max = ct.max(tile, axis=0, keepdims=True)          # (1,) fp32
        local_arg = ct.argmax(tile, axis=0, keepdims=True)       # (1,)
        global_idx = ct.astype(local_arg, np.int64) + j * TILE_N # (1,) int64

        # Strict '>' so the earliest tile that achieves a max wins on ties
        # across tiles (within a single tile, ct.argmax determines tie break).
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

    TILE_N = 2048
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    kernel = _argmax_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, N, TILE_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
