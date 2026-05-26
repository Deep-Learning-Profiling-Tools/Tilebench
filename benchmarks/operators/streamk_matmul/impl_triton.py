"""Triton Stream-K GEMM (Osama et al., PPoPP 2023).

Two-kernel decomposition with the Stream-K scheduler computed INSIDE
each kernel from M/N/K/NUM_SMS/BLOCK_M/BLOCK_N/BLOCK_K. This lets
`triton.autotune` sweep BLOCK_* freely — host code never sees those
values, so there's nothing to keep in sync.

  first_wave  - launches NUM_SMS programs; each owns a contiguous range
                of K-iterations spanning multiple (M, N) output tiles.
                Combines partial tiles via tl.atomic_add into a
                pre-zeroed C — order-independent, no locks needed.
  full_tiles  - data-parallel; one program per leftover whole tile
                after wave-quantization. Writes pre-zeroed disjoint
                regions via tl.store (non-atomic).

Only first_wave is autotuned; full_tiles reuses the winning config so
both kernels share the same BLOCK_* (otherwise their tile partitions
disagree and the result is corrupt).

`tl.dot(..., input_precision="tf32")` is explicit so the precision
choice mirrors cuTile (which casts fp32 → ct.tfloat32 before ct.mma).
"""
import torch
import triton
from triton import language as tl


_DEFAULT_CONFIG = {
    "BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8,
    "num_warps": 8, "num_stages": 4,
}


@triton.jit
def _swizzle_tile(tile_id, M, N,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  GROUP_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


@triton.jit
def first_wave(
    A, B, C, M, N, K, NUM_SMS: tl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    total_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    iters_per_tile = tl.cdiv(K, BLOCK_K)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    total_iters_streamk = streamk_tiles * iters_per_tile
    total_full_iters = total_iters_streamk // NUM_SMS
    total_partial_iters = total_iters_streamk % NUM_SMS

    pid = tl.program_id(0)
    start_iter = pid * total_full_iters + tl.minimum(pid, total_partial_iters)
    last_iter = (pid + 1) * total_full_iters + tl.minimum(pid + 1, total_partial_iters)

    while start_iter < last_iter:
        tile_id = start_iter // iters_per_tile
        rem = iters_per_tile - (start_iter % iters_per_tile)
        end_iter = tl.minimum(start_iter + rem, last_iter)

        pid_m, pid_n = _swizzle_tile(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        mask_m = rm < M
        mask_n = rn < N

        iter_in_tile = start_iter % iters_per_tile
        A_ptrs = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak) \
                 + iter_in_tile * BLOCK_K * stride_ak
        B_ptrs = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn) \
                 + iter_in_tile * BLOCK_K * stride_bk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
        for current_iter in range(start_iter, end_iter):
            mask_k = rk + iter_in_tile * BLOCK_K < K
            a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b, input_precision="tf32")
            A_ptrs += BLOCK_K * stride_ak
            B_ptrs += BLOCK_K * stride_bk
            iter_in_tile += 1

        acc_typed = acc.to(C.dtype.element_ty)
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
        tl.atomic_add(C_, acc_typed, mask=mask_m[:, None] & mask_n[None, :])

        start_iter = end_iter


# 1:1 mirrors impl_cutile.py's _SEARCH_SPACE — bm↔tm, bn↔tn, nw↔occ.
# BLOCK_K and GROUP_M locked to match cuTile's tk=32 / group_m=8.
# `restore_value=["C"]`: first_wave's atomic_add mutates C, so each
# autotune trial must restore C's pre-call state to keep accumulation
# from leaking across trials (and into the post-tune real launch).
_first_wave_autotuned = triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 32, "GROUP_M": 8},
            num_warps=nw, num_stages=3,
        )
        for bm in (64, 128)
        for bn in (128, 256)
        for nw in (4, 8)
    ],
    key=["M", "N", "K"],
    warmup=3,
    rep=10,
    restore_value=["C"],
)(first_wave)


@triton.jit
def full_tiles(
    A, B, C, M, N, K, NUM_SMS: tl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, ACC_TYPE: tl.constexpr,
):
    total_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS

    tile_id = tl.program_id(0) + streamk_tiles
    if tile_id >= total_tiles:
        return

    pid_m, pid_n = _swizzle_tile(tile_id, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    mask_m = rm < M
    mask_n = rn < N

    A_ptrs = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptrs = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = rk + k * BLOCK_K < K
        a = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision="tf32")
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    acc_typed = acc.to(C.dtype.element_ty)
    C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C_, acc_typed, mask=mask_m[:, None] & mask_n[None, :])


def _device_sm_count() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def _streamk_partition(M, N, BLK_M, BLK_N, NUM_SMS):
    """Mirror the kernel's scheduler. Returns (total_tiles, streamk_tiles)."""
    total_tiles = triton.cdiv(M, BLK_M) * triton.cdiv(N, BLK_N)
    streamk_tiles = total_tiles % NUM_SMS
    if total_tiles - streamk_tiles > NUM_SMS:
        streamk_tiles += NUM_SMS
    return total_tiles, streamk_tiles


def run(a: torch.Tensor, b: torch.Tensor,
        block_size: int = None, autotune: bool = False, **kwargs):
    assert a.is_contiguous() and b.is_contiguous()
    assert a.shape[1] == b.shape[0]

    M, K = a.shape
    _, N = b.shape
    NUM_SMS = _device_sm_count()
    ACC_TYPE = tl.float32

    # Mirror impl_cutile.py: route non-fp32 dtypes through an fp32 staging
    # buffer. cuTile *has* to do this (ct.atomic_add doesn't support bf16);
    # Triton matches it so both backends do identical memory traffic and
    # use fp32-precision atomics — otherwise the fp16/bf16 cases would
    # compare Triton's lower-precision fp16 atomic against cuTile's fp32
    # accumulate, plus 2× the C-write bandwidth.
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    if a.dtype == torch.float32:
        c = out
        c.zero_()
    else:
        c = torch.zeros((M, N), device=a.device, dtype=torch.float32)

    common_args = (
        a, b, c, M, N, K, NUM_SMS,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    )

    # ---- Stage 1: first_wave (Stream-K) ----
    if autotune:
        _first_wave_autotuned[(NUM_SMS,)](*common_args, ACC_TYPE=ACC_TYPE)
        chosen = _first_wave_autotuned.best_config
        BLK_M = chosen.kwargs["BLOCK_M"]
        BLK_N = chosen.kwargs["BLOCK_N"]
        BLK_K = chosen.kwargs["BLOCK_K"]
        GROUP_M = chosen.kwargs["GROUP_M"]
        nw, ns = chosen.num_warps, chosen.num_stages
    else:
        cfg = _DEFAULT_CONFIG
        BLK_M, BLK_N, BLK_K = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
        GROUP_M = cfg["GROUP_M"]
        nw, ns = cfg["num_warps"], cfg["num_stages"]
        first_wave[(NUM_SMS,)](
            *common_args,
            BLOCK_M=BLK_M, BLOCK_N=BLK_N, BLOCK_K=BLK_K,
            GROUP_M=GROUP_M, ACC_TYPE=ACC_TYPE,
            num_warps=nw, num_stages=ns,
        )

    # ---- Stage 2: full_tiles (data-parallel) ----
    # Reuses Stage 1's BLK so tile partitioning agrees.
    total_tiles, streamk_tiles = _streamk_partition(M, N, BLK_M, BLK_N, NUM_SMS)
    blocking_tiles = total_tiles - streamk_tiles
    if blocking_tiles > 0:
        full_tiles[(blocking_tiles,)](
            *common_args,
            BLOCK_M=BLK_M, BLOCK_N=BLK_N, BLOCK_K=BLK_K,
            GROUP_M=GROUP_M, ACC_TYPE=ACC_TYPE,
            num_warps=nw, num_stages=ns,
        )

    if c is not out:
        out.copy_(c)
    return out


def get_last_config() -> dict | None:
    cfg = getattr(_first_wave_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_M": cfg.kwargs["BLOCK_M"],
        "BLOCK_N": cfg.kwargs["BLOCK_N"],
        "BLOCK_K": cfg.kwargs["BLOCK_K"],
        "GROUP_M": cfg.kwargs["GROUP_M"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
