import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


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

        tile_val, rel_idx = tl.max(
            vals,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        tile_idx = (n0 + rel_idx).to(tl.int32)
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

    BLOCK_N = 4096
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
