Looking at the trajectory:
- **Iter 2 triton** failed due to a forbidden string ("torch.matmul" in a comment) — easy fix
- **Iter 2 cutile** failed verify on fp32 with 0.001 absolute error (30% mismatch) — classic TF32-precision pattern. Torch reference appears to compute fp32 with IEEE precision, but `ct.mma(fp32,fp32,fp32)` uses TF32 cores.

**Strategy for iter 3 — get verify-clean first:**
- **Triton**: keep iter-1's working stream-K pattern, set `input_precision="ieee"` for fp32 to match torch's accumulation precision; remove any forbidden strings from comments.
- **cuTile**: dispatch fp32 to a separate kernel that casts loaded fp32 → fp64 and uses fp64 mma + fp64 accumulator, then casts to fp32 for the atomic write. This gives precision well above torch's fp32 reference so verify must pass. Keep fp16/bf16 on the existing fast path with split-K = 2.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _streamk_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    iters_per_cta_base, extra_iters, iters_per_tile,
    num_pid_m, num_pid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.assume(iters_per_cta_base >= 0)
    tl.assume(extra_iters >= 0)
    tl.assume(iters_per_tile > 0)
    tl.assume(num_pid_n > 0)
    tl.assume(num_pid_m > 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)

    if pid < extra_iters:
        start_iter = pid * (iters_per_cta_base + 1)
        end_iter = start_iter + iters_per_cta_base + 1
    else:
        start_iter = pid * iters_per_cta_base + extra_iters
        end_iter = start_iter + iters_per_cta_base

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    current_iter = start_iter
    while current_iter < end_iter:
        tile_id = current_iter // iters_per_tile
        tile_iter_start = tile_id * iters_per_tile
        local_start = current_iter - tile_iter_start
        next_tile_iter = tile_iter_start + iters_per_tile
        next_iter = tl.minimum(next_tile_iter, end_iter)
        local_end = next_iter - tile_iter_start

        # L2-friendly swizzled tile ordering
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + offs_m_base
        offs_n = pid_n * BLOCK_N + offs_n_base

        k_off_start = local_start * BLOCK_K
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (k_off_start + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k_off_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn

        m_mask = offs_m[:, None] < M
        n_mask = offs_n[None, :] < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_idx in range(local_start, local_end):
            k_off = k_idx * BLOCK_K
            a_k_mask = (offs_k[None, :] + k_off) < K
            b_k_mask = (offs_k[:, None] + k_off) < K
            a = tl.load(a_ptrs, mask=m_mask & a_k_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_k_mask & n_mask, other=0.0)
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = m_mask & n_mask
        tl.atomic_add(c_ptrs, acc, mask=c_mask)

        current_iter = next_iter


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    # Always accumulate into fp32 buffer so atomic_add stays exact.
    output_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 3 if a.dtype != torch.float32 else 2

    # For fp32, match the reference's IEEE accumulation (no TF32 promotion).
    # For fp16/bf16 the precision arg is a no-op so "ieee" is safe too.
    if a.dtype == torch.float32:
        input_precision = "ieee"
    else:
        input_precision = "tf32"

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    iters_per_tile = triton.cdiv(K, BLOCK_K)
    total_iters = num_pid_m * num_pid_n * iters_per_tile

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_ctas = min(NUM_SMS, total_iters)
    if num_ctas <= 0:
        num_ctas = 1

    iters_per_cta_base = total_iters // num_ctas
    extra_iters = total_iters % num_ctas

    grid = (num_ctas,)
    _streamk_kernel[grid](
        a, b, output_fp32,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output_fp32.stride(0), output_fp32.stride(1),
        iters_per_cta_base, extra_iters, iters_per_tile,
        num_pid_m, num_pid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps, num_stages=num_stages,
    )

    if a.dtype == torch.float32:
        output = output_fp32
    else:
        output = output_fp32.to(a.dtype)

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "num_ctas": num_ctas,
        "input_precision": input_precision,
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


@ct.kernel(occupancy=2)
def _splitk_lowp(a, b, c,
                 ITERS_PER_SPLIT: ConstInt,
                 BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    split_id = ct.bid(2)
    k_base = split_id * ITERS_PER_SPLIT

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k_local in range(0, ITERS_PER_SPLIT):
        k = k_base + k_local
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    ct.atomic_add(c, (offs_m[:, None], offs_n[None, :]), acc)


@ct.kernel(occupancy=1)
def _splitk_fp32_hp(a, b, c,
                    ITERS_PER_SPLIT: ConstInt,
                    BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    split_id = ct.bid(2)
    k_base = split_id * ITERS_PER_SPLIT

    # Promote to fp64 internally: ct.mma(fp32) uses TF32 cores which loses
    # precision vs the reference; fp64 mma gives well-above-fp32 accuracy.
    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float64)
    for k_local in range(0, ITERS_PER_SPLIT):
        k = k_base + k_local
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        a_d = ct.astype(a_tile, np.float64)
        b_d = ct.astype(b_tile, np.float64)
        acc = ct.mma(a_d, b_d, acc)

    acc_f32 = ct.astype(acc, np.float32)
    offs_m = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    ct.atomic_add(c, (offs_m[:, None], offs_n[None, :]), acc_f32)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    if a.dtype == torch.float32:
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        ITERS_PER_SPLIT = 32
        kernel = _splitk_fp32_hp
    else:
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64
        ITERS_PER_SPLIT = 32
        kernel = _splitk_lowp

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    iters_per_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_splits = (iters_per_tile + ITERS_PER_SPLIT - 1) // ITERS_PER_SPLIT
    total_tiles = num_pid_m * num_pid_n

    stream = torch.cuda.current_stream()
    grid = (num_pid_m, num_pid_n, num_splits)
    ct.launch(
        stream, grid, kernel,
        (a, b, output_fp32, ITERS_PER_SPLIT, BLOCK_M, BLOCK_N, BLOCK_K),
    )

    if a.dtype == torch.float32:
        output = output_fp32
    else:
        output = output_fp32.to(a.dtype)

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "ITERS_PER_SPLIT": ITERS_PER_SPLIT,
        "num_splits": num_splits,
        "total_tiles": total_tiles,
        "fp32_path": "fp64_internal" if a.dtype == torch.float32 else "native",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach summary: For Triton, switch fp32 path to `input_precision="ieee"` so `tl.dot` matches the reference's IEEE-754 accumulation (TF32 was causing ~1e-3 absolute errors); remove any forbidden strings. For cuTile, dispatch fp32 to a separate kernel that internally promotes to fp64 and uses fp64 `ct.mma`, side-stepping TF32 precision loss while keeping the existing fp16/bf16 fast path with split-K=2.
