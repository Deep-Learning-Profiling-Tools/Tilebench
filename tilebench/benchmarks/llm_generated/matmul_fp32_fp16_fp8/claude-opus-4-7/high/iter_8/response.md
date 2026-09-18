Looking at the trajectory, the killer problems are:
- **cuTile fp32 at 0.2%** (it's hitting the slow IEEE fp32 path — `ct.mma` with fp32 inputs doesn't auto-use TF32 cores)
- **Triton fp32 at 3.5%** (tf32x3 mainloop with K=20480 needs better pipelining)

The biggest available lever is implementing **manual TF32x3** in cuTile for fp32: decompose `a = a_hi + a_lo` via `tfloat32` casts and issue 3 MMAs per K-block. This drops onto Tensor Cores (~1100 TFLOPS / 3 ≈ 366 effective TFLOPS vs IEEE fp32's 60 TFLOPS ceiling) while preserving fp32-grade precision so verify still passes. For fp16/fp8 keep the best-so-far config.

For Triton fp32, try `BLOCK_K=32, num_stages=4` to deepen the tf32x3 pipeline.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, input_precision=PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)

    if a.dtype == torch.float32:
        # fp32: tf32x3 for precision. Smaller BLOCK_K=32 + stages=4 lets the
        # 3-MMA-per-iter mainloop pipeline more deeply (shmem per stage:
        # 128*32*4*2 = 32KB, ×4 stages = 128KB, well under 228KB on B200).
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        GROUP_SIZE_M = 8
        num_warps = 8
        num_stages = 4
        PRECISION = "tf32x3"
    else:
        # fp16 / fp8: large tile, deep pipeline.
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        GROUP_SIZE_M = 8
        num_warps = 8
        num_stages = 4
        PRECISION = "tf32"

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_kernel[grid](
        a, b, output,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M, PRECISION=PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M, "PRECISION": PRECISION,
        "num_warps": num_warps, "num_stages": num_stages,
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


# fp32 kernel: manual TF32x3 — split each fp32 input into hi (tfloat32) and
# lo (tfloat32 of the residual), then do 3 tensor-core MMAs per K-tile
# (a_hi*b_hi + a_hi*b_lo + a_lo*b_hi). Drops a_lo*b_lo (negligible).
# This recovers fp32-grade precision while running on TF32 tensor cores
# (~1100 TFLOPS) instead of the 60 TFLOPS IEEE fp32 ALU path.
@ct.kernel(occupancy=1)
def _matmul_kernel_tf32x3(a, b, c,
                          K: ConstInt,
                          TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a_tile = ct.load(a, index=(bid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, bid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO)

        a_hi = ct.astype(a_tile, ct.tfloat32)
        a_hi_f = ct.astype(a_hi, np.float32)
        a_lo = ct.astype(a_tile - a_hi_f, ct.tfloat32)

        b_hi = ct.astype(b_tile, ct.tfloat32)
        b_hi_f = ct.astype(b_hi, np.float32)
        b_lo = ct.astype(b_tile - b_hi_f, ct.tfloat32)

        acc = ct.mma(a_lo, b_hi, acc)
        acc = ct.mma(a_hi, b_lo, acc)
        acc = ct.mma(a_hi, b_hi, acc)

    out_tile = ct.astype(acc, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=out_tile)


# fp16 / fp8 kernel: straight MMA, large tile, occ=1 (iter 6 best config).
@ct.kernel(occupancy=1)
def _matmul_kernel_lowp(a, b, c,
                        K: ConstInt,
                        TM: ConstInt, TN: ConstInt, TK: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a_tile = ct.load(a, index=(bid_m, k), shape=(TM, TK),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, bid_n), shape=(TK, TN),
                         padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_tile, b_tile, acc)

    out_tile = ct.astype(acc, c.dtype)
    ct.store(c, index=(bid_m, bid_n), tile=out_tile)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)
    stream = torch.cuda.current_stream()

    if a.dtype == torch.float32:
        # Manual TF32x3 — preserves fp32 precision but uses TF32 tensor cores.
        TM, TN, TK = 128, 128, 32
        kernel = _matmul_kernel_tf32x3
        occupancy = 1
        precision = "tf32x3"
    else:
        TM, TN, TK = 128, 256, 64
        kernel = _matmul_kernel_lowp
        occupancy = 1
        precision = "default"

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, kernel, (a, b, output, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK,
                      "occupancy": occupancy, "precision": precision})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
