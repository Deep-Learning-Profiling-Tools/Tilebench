import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _argmax_rows_masked_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_base = x_ptr + row * stride_m

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)

    best_val = tl.full((), -float("inf"), dtype=tl.float32)
    best_idx = tl.full((), 0, dtype=tl.int32)

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

        tile_val, rel_idx = tl.max(
            vals,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        tile_idx = (n0 + rel_idx).to(tl.int32)
        tile_val_f = tile_val.to(tl.float32)

        take = tile_val_f > best_val
        best_val = tl.where(take, tile_val_f, best_val)
        best_idx = tl.where(take, tile_idx, best_idx)

    tl.store(out_ptr + row, best_idx.to(tl.int64))


@triton.jit
def _argmax_rows_nomask_kernel(
    x_ptr,
    out_ptr,
    N,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    row_base = x_ptr + row * stride_m

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)

    best_val = tl.full((), -float("inf"), dtype=tl.float32)
    best_idx = tl.full((), 0, dtype=tl.int32)

    for n0 in tl.range(0, N, BLOCK_N, num_stages=PIPELINE_STAGES):
        cols = n0 + offs

        vals = tl.load(
            row_base + cols * stride_n,
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
        tile_val_f = tile_val.to(tl.float32)

        take = tile_val_f > best_val
        best_val = tl.where(take, tile_val_f, best_val)
        best_idx = tl.where(take, tile_idx, best_idx)

    tl.store(out_ptr + row, best_idx.to(tl.int64))


@triton.jit
def _argmax_stage1_masked_kernel(
    x_ptr,
    tmp_vals_ptr,
    tmp_idx_ptr,
    N,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)

    offs = tl.arange(0, BLOCK_N)
    n0 = chunk * BLOCK_N
    cols = n0 + offs
    mask = cols < N

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)

    row_base = x_ptr + row * stride_m
    vals = tl.load(
        row_base + cols * stride_n,
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

    tmp_off = row * NUM_CHUNKS + chunk
    tl.store(tmp_vals_ptr + tmp_off, tile_val)
    tl.store(tmp_idx_ptr + tmp_off, tile_idx)


@triton.jit
def _argmax_stage1_nomask_kernel(
    x_ptr,
    tmp_vals_ptr,
    tmp_idx_ptr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    chunk = tl.program_id(0)
    row = tl.program_id(1)

    offs = tl.arange(0, BLOCK_N)
    n0 = chunk * BLOCK_N
    cols = n0 + offs

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)

    row_base = x_ptr + row * stride_m
    vals = tl.load(
        row_base + cols * stride_n,
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

    tmp_off = row * NUM_CHUNKS + chunk
    tl.store(tmp_vals_ptr + tmp_off, tile_val)
    tl.store(tmp_idx_ptr + tmp_off, tile_idx)


@triton.jit
def _argmax_stage2_kernel(
    tmp_vals_ptr,
    tmp_idx_ptr,
    out_ptr,
    NUM_CHUNKS: tl.constexpr,
    FINAL_BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, FINAL_BLOCK_C)
    mask = offs < NUM_CHUNKS

    base = row * NUM_CHUNKS
    vals = tl.load(
        tmp_vals_ptr + base + offs,
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

    best_idx = tl.load(tmp_idx_ptr + base + rel_chunk)
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
    num_warps = 8
    num_stages = 3
    PIPELINE_STAGES = 3
    FINAL_BLOCK_C = 8

    num_chunks = triton.cdiv(N, BLOCK_N)
    FULL_TILE_FAST_PATH = (N % BLOCK_N) == 0
    TWO_STAGE = num_chunks > 1

    if TWO_STAGE:
        tmp_vals = torch.empty((M, num_chunks), device=x.device, dtype=x.dtype)
        tmp_idx = torch.empty((M, num_chunks), device=x.device, dtype=torch.int32)

        grid_stage1 = (num_chunks, M)
        if FULL_TILE_FAST_PATH:
            _argmax_stage1_nomask_kernel[grid_stage1](
                x,
                tmp_vals,
                tmp_idx,
                stride_m=x.stride(0),
                stride_n=x.stride(1),
                NUM_CHUNKS=num_chunks,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            _argmax_stage1_masked_kernel[grid_stage1](
                x,
                tmp_vals,
                tmp_idx,
                N,
                stride_m=x.stride(0),
                stride_n=x.stride(1),
                NUM_CHUNKS=num_chunks,
                BLOCK_N=BLOCK_N,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        _argmax_stage2_kernel[(M,)](
            tmp_vals,
            tmp_idx,
            output,
            NUM_CHUNKS=num_chunks,
            FINAL_BLOCK_C=FINAL_BLOCK_C,
            num_warps=1,
            num_stages=1,
        )
    else:
        grid = (M,)
        if FULL_TILE_FAST_PATH:
            _argmax_rows_nomask_kernel[grid](
                x,
                output,
                N,
                stride_m=x.stride(0),
                stride_n=x.stride(1),
                BLOCK_N=BLOCK_N,
                PIPELINE_STAGES=PIPELINE_STAGES,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            _argmax_rows_masked_kernel[grid](
                x,
                output,
                N,
                stride_m=x.stride(0),
                stride_n=x.stride(1),
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
            "TWO_STAGE": TWO_STAGE,
            "NUM_CHUNKS": num_chunks,
            "FINAL_BLOCK_C": FINAL_BLOCK_C,
            "FINAL_num_warps": 1,
            "FINAL_num_stages": 1,
            "CONST_STRIDES": True,
            "FULL_TILE_FAST_PATH": FULL_TILE_FAST_PATH,
            "CHUNK_TIE_FAST_PATH": True,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
