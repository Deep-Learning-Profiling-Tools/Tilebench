Looking at the trajectory:
- Triton best was iter 1 (53.1%) — basic per-row kernel with `BLOCK_N=4096`
- cuTile best was iter 5 (57.3%) — `TILE_N=2048, occupancy=16`

Both are bandwidth-bound. My hypothesis: processing **multiple rows per CTA** (2D tile loads) will improve memory throughput by amortizing per-CTA overhead and giving TMA/coalescer a larger contiguous request. With M=2048 / ROWS=2 → 1024 CTAs (≈7 per SM on B200), still plenty of parallelism, but each CTA does double the work per iteration.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_kernel(x_ptr, out_ptr, M, N, stride_m,
                   BLOCK_N: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    row_offs = pid * ROWS + tl.arange(0, ROWS)
    row_mask = row_offs < M

    NEG_INF: tl.constexpr = float('-inf')
    best_val = tl.full([ROWS], NEG_INF, dtype=tl.float32)
    best_idx = tl.full([ROWS], 0, dtype=tl.int64)

    for n_start in tl.range(0, N, BLOCK_N, num_stages=3):
        offs = n_start + tl.arange(0, BLOCK_N)
        col_mask = offs < N
        mask = row_mask[:, None] & col_mask[None, :]
        ptrs = x_ptr + row_offs[:, None] * stride_m + offs[None, :]
        x = tl.load(ptrs, mask=mask, other=NEG_INF).to(tl.float32)

        local_max = tl.max(x, axis=1)              # [ROWS]
        local_arg = tl.argmax(x, axis=1)           # [ROWS]
        local_global = local_arg.to(tl.int64) + n_start

        # Strict '>' so earliest occurrence wins on ties across tiles.
        new_better = local_max > best_val
        best_val = tl.where(new_better, local_max, best_val)
        best_idx = tl.where(new_better, local_global, best_idx)

    tl.store(out_ptr + row_offs, best_idx, mask=row_mask)


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    BLOCK_N = 4096
    ROWS = 2
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, ROWS),)
    _argmax_kernel[grid](
        x, output, M, N, x.stride(0),
        BLOCK_N=BLOCK_N, ROWS=ROWS,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_N": BLOCK_N,
        "ROWS": ROWS,
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


@ct.kernel(occupancy=16)
def _argmax_kernel(x, output, N: ConstInt, TILE_N: ConstInt, ROWS: ConstInt):
    bid = ct.bid(0)

    best_val = ct.full((ROWS, 1), -np.inf, dtype=np.float32)
    best_idx = ct.full((ROWS, 1), 0, dtype=np.int64)

    num_tiles = ct.cdiv(N, TILE_N)
    for j in range(num_tiles):
        tile2d = ct.load(
            x, index=(bid, j), shape=(ROWS, TILE_N),
            padding_mode=ct.PaddingMode.NEG_INF,
        )
        tile = ct.astype(tile2d, np.float32)

        local_max = ct.max(tile, axis=1, keepdims=True)
        local_arg = ct.argmax(tile, axis=1, keepdims=True)
        global_idx = ct.astype(local_arg, np.int64) + j * TILE_N

        new_better = local_max > best_val
        best_val = ct.where(new_better, local_max, best_val)
        best_idx = ct.where(new_better, global_idx, best_idx)

    ct.store(output, index=(bid,), tile=best_idx.reshape((ROWS,)))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    assert dim in (1, -1)
    assert x.ndim == 2
    x = x.contiguous()
    M, N = x.shape
    output = torch.empty(M, dtype=torch.int64, device=x.device)

    TILE_N = 2048
    ROWS = 2
    occupancy = 16

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, ROWS), 1, 1)
    ct.launch(stream, grid, _argmax_kernel, (x, output, N, TILE_N, ROWS))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE_N": TILE_N, "ROWS": ROWS, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
