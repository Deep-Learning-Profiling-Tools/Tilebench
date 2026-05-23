Looking at iteration 0 failures:

1. **Triton fp16/bf16**: Numerical precision issues from atomic_add on low-precision types accumulating multiple partial sums.
2. **Triton fp32**: Massive mismatches because PyTorch uses full IEEE fp32 by default (`allow_tf32=False`), but `tl.dot` defaults to TF32.
3. **cuTile all dtypes**: `with_hints()` doesn't exist — hints belong on `@ct.kernel(...)` decorator.

**Fix**: Always accumulate into an fp32 intermediate buffer (atomic_add fp32→fp32 preserves precision), then cast once at the end. For Triton fp32, pass `input_precision="ieee"`. For cuTile, move occupancy hint into `@ct.kernel(occupancy=...)`.

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
    iters_per_cta_base, extra_iters, iters_per_tile, num_pid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    tl.assume(iters_per_cta_base >= 0)
    tl.assume(extra_iters >= 0)
    tl.assume(iters_per_tile > 0)
    tl.assume(num_pid_n > 0)

    if pid < extra_iters:
        start_iter = pid * (iters_per_cta_base + 1)
        end_iter = start_iter + iters_per_cta_base + 1
    else:
        start_iter = pid * iters_per_cta_base + extra_iters
        end_iter = start_iter + iters_per_cta_base

    offs_m_base = tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    current_iter = start_iter
    while current_iter < end_iter:
        tile_id = current_iter // iters_per_tile
        tile_iter_start = tile_id * iters_per_tile
        local_start = current_iter - tile_iter_start
        next_tile_iter = tile_iter_start + iters_per_tile
        next_iter = tl.minimum(next_tile_iter, end_iter)
        local_end = next_iter - tile_iter_start

        pid_m = tile_id // num_pid_n
        pid_n = tile_id % num_pid_n

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
            if USE_IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # Atomic add the fp32 accumulator to the fp32 intermediate buffer.
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = m_mask & n_mask
        tl.atomic_add(c_ptrs, acc, mask=c_mask)

        current_iter = next_iter


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    # Always accumulate into fp32 buffer to avoid precision loss from
    # atomic_add on low-precision dtypes.
    if a.dtype == torch.float32:
        output_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    else:
        output_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    num_warps = 8
    num_stages = 3
    USE_IEEE = (a.dtype == torch.float32)

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
        iters_per_cta_base, extra_iters, iters_per_tile, num_pid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        USE_IEEE=USE_IEEE,
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
        "num_warps": num_warps,
        "num_stages": num_stages,
        "num_ctas": num_ctas,
        "USE_IEEE": USE_IEEE,
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
def _splitk_kernel(a, b, c, num_pid_n,
                   ITERS_PER_SPLIT: ConstInt,
                   BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    tile_id = ct.bid(0)
    split_id = ct.bid(1)

    pid_m = tile_id // num_pid_n
    pid_n = tile_id % num_pid_n

    k_base = split_id * ITERS_PER_SPLIT

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k_local in range(0, ITERS_PER_SPLIT):
        k = k_base + k_local
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    # Atomic add the fp32 accumulator to the fp32 intermediate buffer.
    offs_m_idx = pid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n_idx = pid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    ct.atomic_add(c, (offs_m_idx[:, None], offs_n_idx[None, :]), acc)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    # Always accumulate into fp32 buffer; cast once at the end.
    output_fp32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    ITERS_PER_SPLIT = 32

    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    iters_per_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_splits = (iters_per_tile + ITERS_PER_SPLIT - 1) // ITERS_PER_SPLIT
    total_tiles = num_pid_m * num_pid_n

    stream = torch.cuda.current_stream()
    grid = (total_tiles, num_splits, 1)
    ct.launch(
        stream, grid, _splitk_kernel,
        (a, b, output_fp32, num_pid_n, ITERS_PER_SPLIT, BLOCK_M, BLOCK_N, BLOCK_K),
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
        "occupancy": 2,
        "num_splits": num_splits,
        "total_tiles": total_tiles,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Key changes: (1) Atomic-add into an fp32 staging buffer in both impls — eliminates precision loss on fp16/bf16 paths. (2) Triton fp32 uses `input_precision="ieee"` to match torch's default full-fp32 matmul. (3) cuTile hints moved into the `@ct.kernel(occupancy=2)` decorator (the `with_hints` API doesn't exist).
