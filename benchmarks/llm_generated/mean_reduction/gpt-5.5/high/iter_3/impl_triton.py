import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_reduction_flat_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row_base = row * N

    tl.assume(N > 0)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in tl.range(0, N, BLOCK_N, num_stages=LOOP_STAGES):
        offs = start + cols
        ptrs = x_ptr + row_base + offs
        if EVEN_N:
            vals = tl.load(
                ptrs,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            mask = offs < N
            vals = tl.load(
                ptrs,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc += vals

    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total * (1.0 / N))


@triton.jit
def _mean_reduction_strided_kernel(
    x_ptr,
    out_ptr,
    stride_m,
    stride_n,
    N,
    BLOCK_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)
    tl.assume(N > 0)

    row_ptr = x_ptr + row * stride_m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in tl.range(0, N, BLOCK_N, num_stages=LOOP_STAGES):
        offs = start + cols
        ptrs = row_ptr + offs * stride_n
        if EVEN_N:
            vals = tl.load(
                ptrs,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            mask = offs < N
            vals = tl.load(
                ptrs,
                mask=mask,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc += vals

    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, total * (1.0 / N))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_N = 1024
    num_warps = 8
    num_stages = 3
    LOOP_STAGES = 3
    EVEN_N = (N % BLOCK_N) == 0
    USE_FLAT = x.is_contiguous()

    if USE_FLAT:
        _mean_reduction_flat_kernel[(M,)](
            x,
            output,
            N,
            BLOCK_N=BLOCK_N,
            EVEN_N=EVEN_N,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _mean_reduction_strided_kernel[(M,)](
            x,
            output,
            x.stride(0),
            x.stride(1),
            N,
            BLOCK_N=BLOCK_N,
            EVEN_N=EVEN_N,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOOP_STAGES": LOOP_STAGES,
            "EVEN_N": EVEN_N,
            "USE_FLAT": USE_FLAT,
            "cache_modifier": ".cg",
            "eviction_policy": "evict_first",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
