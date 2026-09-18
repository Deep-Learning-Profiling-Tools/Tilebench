import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_reduction_pair_indep_flat_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_N: tl.constexpr,
    PAIR_N: tl.constexpr,
    EVEN_PAIR: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    base = row * N

    tl.assume(N > 0)

    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in tl.range(0, N, PAIR_N, num_stages=LOOP_STAGES):
        offs0 = start + cols
        if EVEN_PAIR:
            vals0 = tl.load(
                x_ptr + base + offs0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals0 = tl.load(
                x_ptr + base + offs0,
                mask=offs0 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc0 += vals0

        offs1 = start + BLOCK_N + cols
        if EVEN_PAIR:
            vals1 = tl.load(
                x_ptr + base + offs1,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals1 = tl.load(
                x_ptr + base + offs1,
                mask=offs1 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc1 += vals1

    total = tl.sum(acc0 + acc1, axis=0)
    tl.store(out_ptr + row, total * (1.0 / N))


@triton.jit
def _mean_reduction_pair_indep_strided_kernel(
    x_ptr,
    out_ptr,
    stride_m,
    stride_n,
    N,
    BLOCK_N: tl.constexpr,
    PAIR_N: tl.constexpr,
    EVEN_PAIR: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)
    tl.assume(N > 0)

    row_ptr = x_ptr + row * stride_m

    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in tl.range(0, N, PAIR_N, num_stages=LOOP_STAGES):
        offs0 = start + cols
        ptrs0 = row_ptr + offs0 * stride_n
        if EVEN_PAIR:
            vals0 = tl.load(
                ptrs0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals0 = tl.load(
                ptrs0,
                mask=offs0 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc0 += vals0

        offs1 = start + BLOCK_N + cols
        ptrs1 = row_ptr + offs1 * stride_n
        if EVEN_PAIR:
            vals1 = tl.load(
                ptrs1,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals1 = tl.load(
                ptrs1,
                mask=offs1 < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc1 += vals1

    total = tl.sum(acc0 + acc1, axis=0)
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
    PAIR_UNROLL = 2
    PAIR_N = BLOCK_N * PAIR_UNROLL
    num_warps = 4
    num_stages = 3
    LOOP_STAGES = 3
    EVEN_PAIR = (N % PAIR_N) == 0
    USE_FLAT = x.is_contiguous()

    if USE_FLAT:
        _mean_reduction_pair_indep_flat_kernel[(M,)](
            x,
            output,
            N,
            BLOCK_N=BLOCK_N,
            PAIR_N=PAIR_N,
            EVEN_PAIR=EVEN_PAIR,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _mean_reduction_pair_indep_strided_kernel[(M,)](
            x,
            output,
            x.stride(0),
            x.stride(1),
            N,
            BLOCK_N=BLOCK_N,
            PAIR_N=PAIR_N,
            EVEN_PAIR=EVEN_PAIR,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ALGO": "flat_pair_indep2" if USE_FLAT else "strided_pair_indep2",
            "BLOCK_N": BLOCK_N,
            "PAIR_N": PAIR_N,
            "PAIR_UNROLL": PAIR_UNROLL,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOOP_STAGES": LOOP_STAGES,
            "EVEN_PAIR": EVEN_PAIR,
            "USE_FLAT": USE_FLAT,
            "cache_modifier": ".cg",
            "eviction_policy": "evict_first",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
