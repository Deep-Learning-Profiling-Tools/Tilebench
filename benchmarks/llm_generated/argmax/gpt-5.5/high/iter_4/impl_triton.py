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
