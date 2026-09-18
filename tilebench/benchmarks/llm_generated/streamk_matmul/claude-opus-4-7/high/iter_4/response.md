Looking at the trajectory, the key insight is that **stream-K's atomic_add is destroying fp32 precision**. With `atol=1e-5/rtol=1.3e-6` defaults being applied (despite config saying 1.0/1e-2), atomic ordering introduces ~0.001 error which is way over tolerance.

**Strategy for iter 4:**
1. **Switch from atomic stream-K to persistent data-parallel scheduling** — each CTA owns whole output tiles, no atomics, deterministic K-reduction
2. **Triton fp32**: use `input_precision="tf32x3"` (3xTF32 emulation, near-fp32 accuracy, ~366 TFLOPS effective vs 75 TFLOPS for IEEE)
3. **cuTile fp32**: manually implement 3xTF32 emulation via `ct.astype(x, ct.tfloat32)` + residual splitting
4. Keep tile config 128×128×64 with num_warps=8, num_stages=3 — fits all dtypes' shared memory budgets

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_persistent_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    total_tiles, num_pid_m, num_pid_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_start = tl.program_id(0)
    num_progs = tl.num_programs(0)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    iters_per_tile = tl.cdiv(K, BLOCK_K)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(num_pid_m > 0)
    tl.assume(num_pid_n > 0)

    # Persistent loop: each CTA processes multiple output tiles.
    # Each tile is owned by exactly one CTA -> no atomics -> deterministic.
    for tile_id in tl.range(pid_start, total_tiles, num_progs, num_stages=1):
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(iters_per_tile):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
            acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 3

    # For fp32: use 3xTF32 emulation — near-fp32 accuracy (~10^-7) at ~1/3 TF32 perf,
    # vastly faster than pure IEEE (which uses CUDA cores at ~75 TFLOPS on B200).
    if a.dtype == torch.float32:
        input_precision = "tf32x3"
    else:
        input_precision = "tf32"

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    total_tiles = num_pid_m * num_pid_n

    NUM_SMS = torch.cuda.get_device_properties(a.device).multi_processor_count
    num_ctas = min(NUM_SMS, total_tiles)
    if num_ctas <= 0:
        num_ctas = 1

    grid = (num_ctas,)
    _matmul_persistent_kernel[grid](
        a, b, output,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        total_tiles, num_pid_m, num_pid_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        INPUT_PRECISION=input_precision,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_warps": num_warps, "num_stages": num_stages,
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
def _matmul_lowp(a, b, c, K_TILES: ConstInt,
                 BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k in range(K_TILES):
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    result = ct.astype(acc, c.dtype)
    ct.store(c, index=(pid_m, pid_n), tile=result)


@ct.kernel(occupancy=2)
def _matmul_fp32_3xtf32(a, b, c, K_TILES: ConstInt,
                        BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)
    for k in range(K_TILES):
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)

        # 3xTF32 emulation: split fp32 into hi (truncated to tf32 precision)
        # and lo (residual, also tf32-representable). Then 3 mmas approximate
        # fp32 to ~10^-7 absolute, well within fp32 tolerance.
        a_hi = ct.astype(a_tile, ct.tfloat32)
        a_hi_f = ct.astype(a_hi, np.float32)
        a_lo = ct.astype(a_tile - a_hi_f, ct.tfloat32)

        b_hi = ct.astype(b_tile, ct.tfloat32)
        b_hi_f = ct.astype(b_hi, np.float32)
        b_lo = ct.astype(b_tile - b_hi_f, ct.tfloat32)

        acc = ct.mma(a_hi, b_hi, acc)
        acc = ct.mma(a_hi, b_lo, acc)
        acc = ct.mma(a_lo, b_hi, acc)

    ct.store(c, index=(pid_m, pid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    K_TILES = (K + BLOCK_K - 1) // BLOCK_K
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N

    stream = torch.cuda.current_stream()
    grid = (num_pid_m, num_pid_n, 1)

    if a.dtype == torch.float32:
        kernel = _matmul_fp32_3xtf32
        kernel_name = "fp32_3xtf32"
    else:
        kernel = _matmul_lowp
        kernel_name = "lowp"

    ct.launch(stream, grid, kernel,
              (a, b, output, K_TILES, BLOCK_M, BLOCK_N, BLOCK_K))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "K_TILES": K_TILES,
        "num_pid_m": num_pid_m, "num_pid_n": num_pid_n,
        "kernel": kernel_name,
        "occupancy": 2,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

**Approach summary**: Drop atomic-stream-K (it was the source of fp32 verification failures). Switch to persistent data-parallel scheduling where each CTA owns whole output tiles — deterministic K-reduction, no atomic ordering noise. For fp32 accuracy, use Triton's built-in `tf32x3` precision mode, and emulate the same 3xTF32 trick manually in cuTile via `ct.tfloat32` casts + residual splitting.
