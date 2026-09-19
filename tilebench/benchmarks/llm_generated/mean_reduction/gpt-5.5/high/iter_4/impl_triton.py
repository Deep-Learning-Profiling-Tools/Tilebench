import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _mean_reduction_split_partial_kernel(
    x_ptr,
    partials_ptr,
    stride_m,
    stride_n,
    N,
    BLOCK_N: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    EVEN_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    part = tl.program_id(1)
    cols = tl.arange(0, BLOCK_N)

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)
    tl.assume(N > 0)

    num_tiles = tl.cdiv(N, BLOCK_N)
    tiles_per_part = tl.cdiv(num_tiles, NUM_PARTS)
    start_tile = part * tiles_per_part
    end_tile = tl.minimum(start_tile + tiles_per_part, num_tiles)

    row_ptr = x_ptr + row * stride_m
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for tile_idx in tl.range(start_tile, end_tile, 1, num_stages=LOOP_STAGES):
        offs = tile_idx * BLOCK_N + cols
        ptrs = row_ptr + offs * stride_n
        if EVEN_N:
            vals = tl.load(
                ptrs,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        else:
            vals = tl.load(
                ptrs,
                mask=offs < N,
                other=0.0,
                cache_modifier=".cg",
                eviction_policy="evict_first",
            ).to(tl.float32)
        acc += vals

    total = tl.sum(acc, axis=0)
    tl.store(partials_ptr + row * NUM_PARTS + part, total)


@triton.jit
def _mean_reduction_mma_partial_kernel(
    x_ptr,
    partials_ptr,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    NUM_PARTS: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    part = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    tl.assume(stride_m > 0)
    tl.assume(stride_n > 0)
    tl.assume(M > 0)
    tl.assume(N > 0)

    num_tiles = tl.cdiv(N, BLOCK_K)
    tiles_per_part = tl.cdiv(num_tiles, NUM_PARTS)
    start_tile = part * tiles_per_part
    end_tile = tl.minimum(start_tile + tiles_per_part, num_tiles)

    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    for tile_idx in tl.range(start_tile, end_tile, 1, num_stages=LOOP_STAGES):
        k = tile_idx * BLOCK_K + offs_k
        ptrs = x_ptr + rows[:, None] * stride_m + k[None, :] * stride_n
        mask = (rows[:, None] < M) & (k[None, :] < N)
        a = tl.load(
            ptrs,
            mask=mask,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        ones = tl.full((BLOCK_K, BLOCK_R), 1.0, dtype=tl.float32).to(a.dtype)
        acc = tl.dot(a, ones, acc, out_dtype=tl.float32)

    partial = tl.sum(acc, axis=1) * (1.0 / BLOCK_R)
    tl.store(partials_ptr + rows * NUM_PARTS + part, partial, mask=rows < M)


@triton.jit
def _mean_reduction_finalize_kernel(
    partials_ptr,
    out_ptr,
    N,
    NUM_PARTS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_P)
    mask = offs < NUM_PARTS

    vals = tl.load(partials_ptr + row * NUM_PARTS + offs, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    tl.store(out_ptr + row, total * (1.0 / N))


def run(x: torch.Tensor, dim: int = 1, **kwargs) -> torch.Tensor:
    if dim < 0:
        dim = dim + x.dim()
    if dim != 1:
        raise NotImplementedError("mean_reduction only supports row-wise dim=1")

    M = x.shape[0]
    N = x.shape[1]
    output = torch.empty((M,), device=x.device, dtype=torch.float32)

    NUM_PARTS = 8
    BLOCK_P = 8
    LOOP_STAGES = 3

    # fp16/bf16 path: reduce as a skinny GEMM against an implicit all-ones matrix,
    # using Tensor Cores for the expensive inner reduction.
    DOT_BLOCK_M = 16
    DOT_BLOCK_K = 128
    DOT_BLOCK_R = 8
    DOT_NUM_WARPS = 4
    DOT_NUM_STAGES = 3

    # fp32 path: keep IEEE fp32 accumulation and improve parallelism by splitting
    # each row over several CTAs, followed by a tiny final reduction.
    RED_BLOCK_N = 1024
    RED_NUM_WARPS = 4
    RED_NUM_STAGES = 3
    EVEN_N = (N % RED_BLOCK_N) == 0

    USE_MMA = x.dtype != torch.float32
    partials = torch.empty((M, NUM_PARTS), device=x.device, dtype=torch.float32)

    if USE_MMA:
        grid_partial = (triton.cdiv(M, DOT_BLOCK_M), NUM_PARTS)
        _mean_reduction_mma_partial_kernel[grid_partial](
            x,
            partials,
            M,
            N,
            x.stride(0),
            x.stride(1),
            BLOCK_M=DOT_BLOCK_M,
            BLOCK_K=DOT_BLOCK_K,
            BLOCK_R=DOT_BLOCK_R,
            NUM_PARTS=NUM_PARTS,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=DOT_NUM_WARPS,
            num_stages=DOT_NUM_STAGES,
        )
    else:
        grid_partial = (M, NUM_PARTS)
        _mean_reduction_split_partial_kernel[grid_partial](
            x,
            partials,
            x.stride(0),
            x.stride(1),
            N,
            BLOCK_N=RED_BLOCK_N,
            NUM_PARTS=NUM_PARTS,
            EVEN_N=EVEN_N,
            LOOP_STAGES=LOOP_STAGES,
            num_warps=RED_NUM_WARPS,
            num_stages=RED_NUM_STAGES,
        )

    _mean_reduction_finalize_kernel[(M,)](
        partials,
        output,
        N,
        NUM_PARTS=NUM_PARTS,
        BLOCK_P=BLOCK_P,
        num_warps=1,
        num_stages=1,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "ALGO": "hybrid_mma_f16bf16_split_fp32",
            "USE_MMA": USE_MMA,
            "NUM_PARTS": NUM_PARTS,
            "BLOCK_P": BLOCK_P,
            "LOOP_STAGES": LOOP_STAGES,
            "DOT_BLOCK_M": DOT_BLOCK_M,
            "DOT_BLOCK_K": DOT_BLOCK_K,
            "DOT_BLOCK_R": DOT_BLOCK_R,
            "DOT_num_warps": DOT_NUM_WARPS,
            "DOT_num_stages": DOT_NUM_STAGES,
            "RED_BLOCK_N": RED_BLOCK_N,
            "RED_num_warps": RED_NUM_WARPS,
            "RED_num_stages": RED_NUM_STAGES,
            "EVEN_N": EVEN_N,
            "cache_modifier": ".cg",
            "eviction_policy": "evict_first",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
