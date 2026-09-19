```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_pair_combine(v1, i1, v2, i2):
    take_v2 = (v2 > v1) | ((v2 == v1) & (i2 < i1))
    return tl.where(take_v2, v2, v1), tl.where(take_v2, i2, i1)


@triton.jit
def _argmax_rows_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_m,
    stride_n,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_base = x_ptr + row * stride_m

    best_val = tl.full((), -float("inf"), dtype=tl.float32)
    best_idx = tl.full((), 2147483647, dtype=tl.int32)

    for n0 in tl.range(0, N, BLOCK_N, num_stages=PIPELINE_STAGES):
        cols = n0 + offs
        mask = cols < N

        vals = tl.load(
            row_base + cols * stride_n,
            mask=mask,
            other=-float("inf"),
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        idxs = tl.where(mask, cols, 2147483647).to(tl.int32)

        tile_val, tile_idx = tl.reduce(
            (vals, idxs),
            axis=0,
            combine_fn=_argmax_pair_combine,
        )
        tile_val_f = tile_val.to(tl.float32)

        take = (tile_val_f > best_val) | ((tile_val_f == best_val) & (tile_idx < best_idx))
        best_val = tl.where(take, tile_val_f, best_val)
        best_idx = tl.where(take, tile_idx, best_idx)

    tl.store(out_ptr + row, best_idx.to(tl.int64))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim == -1:
        dim = 1
    if dim != 1:
        raise NotImplementedError("This TileBench argmax implementation supports row-wise dim=1 only.")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.int64)

    BLOCK_N = 2048
    num_warps = 8
    num_stages = 3
    PIPELINE_STAGES = 3

    grid = (M,)
    _argmax_rows_kernel[grid](
        x,
        output,
        N,
        x.stride(0),
        x.stride(1),
        BLOCK_N=BLOCK_N,
        PIPELINE_STAGES=PIPELINE_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_N": BLOCK_N,
            "PIPELINE_STAGES": PIPELINE_STAGES,
            "num_warps": num_warps,
            "num_stages": num_stages,
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

_LAST_CFG: dict = {}


@ct.function
def _argmax_pair_combine(v1, i1, v2, i2):
    take_v2 = (v2 > v1) | ((v2 == v1) & (i2 < i1))
    return ct.where(take_v2, v2, v1), ct.where(take_v2, i2, i1)


@ct.kernel
def _argmax_rows_kernel(x, output, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    offs = ct.arange(TILE, dtype=np.int32)

    best_val = ct.full((1,), -np.inf, dtype=np.float32)
    best_idx = ct.full((1,), 2147483647, dtype=np.int32)

    num_tiles = ct.cdiv(N, TILE)
    for j in range(0, num_tiles):
        vals = ct.load(
            x,
            index=(row, j),
            shape=(1, TILE),
            padding_mode=ct.PaddingMode.NEG_INF,
            latency=1,
            allow_tma=False,
        )

        idxs = offs[None, :] + j * TILE
        valid = idxs < N
        idxs = ct.where(valid, idxs, 2147483647)

        tile_val, tile_idx = ct.reduce(
            (vals, idxs),
            axis=1,
            func=_argmax_pair_combine,
            identity=(-np.inf, 2147483647),
            keepdims=False,
        )
        tile_val_f = ct.astype(tile_val, np.float32)

        take = (tile_val_f > best_val) | ((tile_val_f == best_val) & (tile_idx < best_idx))
        best_val = ct.where(take, tile_val_f, best_val)
        best_idx = ct.where(take, tile_idx, best_idx)

    ct.store(output, index=(row,), tile=ct.astype(best_idx, output.dtype))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim == -1:
        dim = 1
    if dim != 1:
        raise NotImplementedError("This TileBench argmax implementation supports row-wise dim=1 only.")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.int64)
    stream = torch.cuda.current_stream()

    TILE = 2048
    occupancy = 4

    grid = (M, 1, 1)
    kernel = _argmax_rows_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, N, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
