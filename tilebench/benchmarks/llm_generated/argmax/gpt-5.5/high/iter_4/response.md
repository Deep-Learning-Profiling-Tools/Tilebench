```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_partials_masked_kernel(
    x_ptr,
    tmp_vals_ptr,
    tmp_idx_ptr,
    N,
    tmp_stride,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    offs = tl.arange(0, BLOCK_N)
    n0 = chunk * BLOCK_N
    cols = n0 + offs
    mask = cols < N

    vals = tl.load(
        x_ptr + row * stride_m + cols * stride_n,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    tile_val, rel_idx = tl.max(
        vals,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    tile_idx = (n0 + rel_idx).to(tl.int32)

    out_off = row * tmp_stride + chunk
    tl.store(tmp_vals_ptr + out_off, tile_val)
    tl.store(tmp_idx_ptr + out_off, tile_idx)


@triton.jit
def _argmax_partials_nomask_kernel(
    x_ptr,
    tmp_vals_ptr,
    tmp_idx_ptr,
    N,
    tmp_stride,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)

    offs = tl.arange(0, BLOCK_N)
    n0 = chunk * BLOCK_N
    cols = n0 + offs

    vals = tl.load(
        x_ptr + row * stride_m + cols * stride_n,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    tile_val, rel_idx = tl.max(
        vals,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    tile_idx = (n0 + rel_idx).to(tl.int32)

    out_off = row * tmp_stride + chunk
    tl.store(tmp_vals_ptr + out_off, tile_val)
    tl.store(tmp_idx_ptr + out_off, tile_idx)


@triton.jit
def _argmax_finalize_kernel(
    tmp_vals_ptr,
    tmp_idx_ptr,
    out_ptr,
    num_chunks,
    tmp_stride,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < num_chunks

    vals = tl.load(
        tmp_vals_ptr + row * tmp_stride + offs,
        mask=mask,
        other=-float("inf"),
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    _, rel_chunk = tl.max(
        vals,
        axis=0,
        return_indices=True,
        return_indices_tie_break_left=True,
    )
    best_idx = tl.load(tmp_idx_ptr + row * tmp_stride + rel_chunk)
    tl.store(out_ptr + row, best_idx.to(tl.int64))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim == -1:
        dim = 1
    if dim != 1:
        raise NotImplementedError("This TileBench argmax implementation supports row-wise dim=1 only.")

    M = x.shape[0]
    N = x.shape[1]

    output = torch.empty((M,), device=x.device, dtype=torch.int64)

    BLOCK_N = 4096
    BLOCK_C = 8
    partial_num_warps = 8
    partial_num_stages = 3
    final_num_warps = 1
    final_num_stages = 1

    num_chunks = triton.cdiv(N, BLOCK_N)
    full_tile_fast_path = (N % BLOCK_N) == 0

    tmp_vals = torch.empty((M, num_chunks), device=x.device, dtype=x.dtype)
    tmp_idx = torch.empty((M, num_chunks), device=x.device, dtype=torch.int32)

    partial_grid = (M, num_chunks)
    if full_tile_fast_path:
        _argmax_partials_nomask_kernel[partial_grid](
            x,
            tmp_vals,
            tmp_idx,
            N,
            num_chunks,
            stride_m=x.stride(0),
            stride_n=x.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=partial_num_warps,
            num_stages=partial_num_stages,
        )
    else:
        _argmax_partials_masked_kernel[partial_grid](
            x,
            tmp_vals,
            tmp_idx,
            N,
            num_chunks,
            stride_m=x.stride(0),
            stride_n=x.stride(1),
            BLOCK_N=BLOCK_N,
            num_warps=partial_num_warps,
            num_stages=partial_num_stages,
        )

    _argmax_finalize_kernel[(M,)](
        tmp_vals,
        tmp_idx,
        output,
        num_chunks,
        num_chunks,
        BLOCK_C=BLOCK_C,
        num_warps=final_num_warps,
        num_stages=final_num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TWO_STAGE": True,
            "BLOCK_N": BLOCK_N,
            "BLOCK_C": BLOCK_C,
            "NUM_CHUNKS": num_chunks,
            "partial_num_warps": partial_num_warps,
            "partial_num_stages": partial_num_stages,
            "final_num_warps": final_num_warps,
            "final_num_stages": final_num_stages,
            "CONST_STRIDES": True,
            "FULL_TILE_FAST_PATH": full_tile_fast_path,
            "CHUNK_TIE_FAST_PATH": True,
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
def _argmax_partials_nomask_kernel(x, tmp_vals, tmp_idx, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    chunk = ct.bid(1)

    offs = ct.arange(TILE, dtype=np.int32)
    vals = ct.load(
        x,
        index=(row, chunk),
        shape=(1, TILE),
        latency=1,
        allow_tma=False,
    )
    idxs = offs[None, :] + chunk * TILE

    tile_val, tile_idx = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_pair_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    ct.store(tmp_vals, index=(row, chunk), tile=tile_val, latency=1, allow_tma=False)
    ct.store(tmp_idx, index=(row, chunk), tile=tile_idx, latency=1, allow_tma=False)


@ct.kernel
def _argmax_partials_masked_kernel(x, tmp_vals, tmp_idx, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    chunk = ct.bid(1)

    offs = ct.arange(TILE, dtype=np.int32)
    vals = ct.load(
        x,
        index=(row, chunk),
        shape=(1, TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )

    idxs = offs[None, :] + chunk * TILE
    valid = idxs < N
    idxs = ct.where(valid, idxs, 2147483647)

    tile_val, tile_idx = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_pair_combine,
        identity=(-np.inf, 2147483647),
        keepdims=True,
    )

    ct.store(tmp_vals, index=(row, chunk), tile=tile_val, latency=1, allow_tma=False)
    ct.store(tmp_idx, index=(row, chunk), tile=tile_idx, latency=1, allow_tma=False)


@ct.kernel
def _argmax_finalize_kernel(tmp_vals, tmp_idx, output, NUM_CHUNKS: ConstInt, CHUNK_TILE: ConstInt):
    row = ct.bid(0)

    offs = ct.arange(CHUNK_TILE, dtype=np.int32)
    vals = ct.load(
        tmp_vals,
        index=(row, 0),
        shape=(1, CHUNK_TILE),
        padding_mode=ct.PaddingMode.NEG_INF,
        latency=1,
        allow_tma=False,
    )
    idxs = ct.load(
        tmp_idx,
        index=(row, 0),
        shape=(1, CHUNK_TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    valid = offs[None, :] < NUM_CHUNKS
    idxs = ct.where(valid, idxs, 2147483647)

    _, best_idx = ct.reduce(
        (vals, idxs),
        axis=1,
        func=_argmax_pair_combine,
        identity=(-np.inf, 2147483647),
        keepdims=False,
    )

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

    TILE = 4096
    CHUNK_TILE = 8
    partial_occupancy = 4
    final_occupancy = 8

    num_chunks = ct.cdiv(N, TILE)
    full_tile_fast_path = (N % TILE) == 0

    tmp_vals = torch.empty((M, num_chunks), device=x.device, dtype=x.dtype)
    tmp_idx = torch.empty((M, num_chunks), device=x.device, dtype=torch.int32)

    partial_grid = (M, num_chunks, 1)
    if full_tile_fast_path:
        partial_kernel = _argmax_partials_nomask_kernel.with_hints(occupancy=partial_occupancy)
        ct.launch(stream, partial_grid, partial_kernel, (x, tmp_vals, tmp_idx, N, TILE))
    else:
        partial_kernel = _argmax_partials_masked_kernel.with_hints(occupancy=partial_occupancy)
        ct.launch(stream, partial_grid, partial_kernel, (x, tmp_vals, tmp_idx, N, TILE))

    final_kernel = _argmax_finalize_kernel.with_hints(occupancy=final_occupancy)
    ct.launch(stream, (M, 1, 1), final_kernel, (tmp_vals, tmp_idx, output, num_chunks, CHUNK_TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TWO_STAGE": True,
            "TILE": TILE,
            "CHUNK_TILE": CHUNK_TILE,
            "NUM_CHUNKS": num_chunks,
            "partial_occupancy": partial_occupancy,
            "final_occupancy": final_occupancy,
            "FULL_TILE_FAST_PATH": full_tile_fast_path,
            "CHUNK_TIE_FAST_PATH": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
