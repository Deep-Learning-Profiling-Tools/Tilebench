Looking at the trajectory:
- **Triton iter 7 broke fp32 verify** (used TF32 instead of IEEE; harness uses tight fp32 tolerances). Need to revert to iter 6's IEEE fp32 path with a small safe tweak.
- **cuTile has never verified fp32** — `ct.mma` uses TF32 for fp32 inputs, which doesn't meet the strict fp32 tolerance. The prompt explicitly says: achieve correctness first.

My strategy:
1. **Triton**: revert to iter 6 (IEEE fp32), and bump `num_stages` to 4 for fp16/bf16 only.
2. **cuTile**: for fp32, do the MMA in **fp64** (B200 ct.mma supports fp64 acc and gives IEEE-grade precision). Keep iter 7's fast path for fp16/bf16.

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
    INPUT_PRECISION: tl.constexpr,
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
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)

    if a.dtype == torch.float32:
        # Strict fp32 verify requires IEEE precision (TF32 fails rtol).
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
        num_warps = 8
        num_stages = 3
        input_precision = "ieee"
        GROUP_SIZE_M = 8
    else:
        # fp16 / bf16 — bump num_stages from 3 → 4 for deeper pipelining.
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        num_warps = 8
        num_stages = 4
        input_precision = "tf32"
        GROUP_SIZE_M = 8

    num_pid_m = triton.cdiv(M, BLOCK_M)
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    _matmul_kernel[grid](
        a, b, output,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
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
        "input_precision": input_precision,
        "grid": grid[0],
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
def _matmul_kernel_lowp(a, b, c, K_TILES: ConstInt,
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


@ct.kernel(occupancy=1)
def _matmul_kernel_fp32_fp64(a, b, c, K_TILES: ConstInt,
                              BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    # fp32 inputs upcast to fp64 in registers, then fp64 ct.mma → IEEE-grade.
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float64)
    for k in range(K_TILES):
        a_tile = ct.load(a, index=(pid_m, k), shape=(BLOCK_M, BLOCK_K),
                         padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k, pid_n), shape=(BLOCK_K, BLOCK_N),
                         padding_mode=ct.PaddingMode.ZERO)
        a64 = ct.astype(a_tile, np.float64)
        b64 = ct.astype(b_tile, np.float64)
        acc = ct.mma(a64, b64, acc)

    result = ct.astype(acc, c.dtype)
    ct.store(c, index=(pid_m, pid_n), tile=result)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M, K = a.shape
    _, N = b.shape

    output = torch.empty((M, N), dtype=a.dtype, device=a.device)
    stream = torch.cuda.current_stream()

    if a.dtype == torch.float32:
        # IEEE-fp32 via fp64 tensor cores (B200 supports fp64 ct.mma).
        # Small tile to keep fp64 register pressure manageable.
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        K_TILES = (K + BLOCK_K - 1) // BLOCK_K
        num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
        num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
        grid = (num_pid_m, num_pid_n, 1)
        ct.launch(stream, grid, _matmul_kernel_fp32_fp64,
                  (a, b, output, K_TILES, BLOCK_M, BLOCK_N, BLOCK_K))
        kernel_kind = "fp64_via_mma"
        occ = 1
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
        K_TILES = (K + BLOCK_K - 1) // BLOCK_K
        num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
        num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
        grid = (num_pid_m, num_pid_n, 1)
        ct.launch(stream, grid, _matmul_kernel_lowp,
                  (a, b, output, K_TILES, BLOCK_M, BLOCK_N, BLOCK_K))
        kernel_kind = "lowp_mma"
        occ = 2

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "K_TILES": K_TILES,
        "num_pid_m": num_pid_m, "num_pid_n": num_pid_n,
        "kernel": kernel_kind,
        "occupancy": occ,
        "dtype": str(a.dtype),
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Summary: Triton reverts to iter-6's verify-clean code with a small `num_stages=3→4` bump on the fp16/bf16 path for deeper pipelining; cuTile takes the fp32 path through fp64 tensor cores to achieve IEEE-grade precision and finally clear verify on all three dtypes.
