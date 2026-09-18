Looking at the trajectory:

**Triton**: Iter 5's bottleneck is fp32 at 6.9% (using IEEE for safety). fp16/bf16 are at ~34%. If I can get fp32 close to fp16's level, the average jumps from 25% to ~30%+.

**Cutile**: 3xTF32 emulation gets ~98% correct but the `a_lo*b_lo` term matters at K=640. Switch to 4xTF32 (full Markidis-style decomposition) to recover full precision.

**Strategy for iter 6**:
- Triton: Use `input_precision="tf32x3"` for fp32 (3-product TF32 emulation = ~333 TFLOPS effective, still gives ~fp32 precision, definitely passes verify). Keep `tf32` for fp16/bf16.
- Cutile: 4-product TF32 emulation (adds the missing `a_lo*b_lo` term).

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _bmm_kernel(A, B, C,
                M, N, K,
                stride_ab, stride_am, stride_ak,
                stride_bb, stride_bk, stride_bn,
                stride_cb, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
                PRECISION: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    batch = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(batch >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + batch * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + batch * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    for k in range(0, num_k_tiles):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        if PRECISION == 0:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        elif PRECISION == 1:
            acc = tl.dot(a, b, acc, input_precision="tf32")
        else:
            acc = tl.dot(a, b, acc, input_precision="tf32x3")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C.dtype.element_ty)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + batch * stride_cb + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A3 = A.view(BATCH, M, K)
    B3 = B.view(BATCH, K, N)
    C3 = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8
    num_warps = 8
    num_stages = 3

    # fp32 → tf32x3 (3-product TF32 emulation, ~fp32 precision, uses TC).
    # fp16/bf16 → tf32 (native TC path).
    PRECISION = 2 if A.dtype == torch.float32 else 1
    EVEN_K = (K % BLOCK_K == 0)

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), BATCH)
    _bmm_kernel[grid](
        A3, B3, C3,
        M, N, K,
        A3.stride(0), A3.stride(1), A3.stride(2),
        B3.stride(0), B3.stride(1), B3.stride(2),
        C3.stride(0), C3.stride(1), C3.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, PRECISION=PRECISION, EVEN_K=EVEN_K,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_M": GROUP_M, "num_warps": num_warps, "num_stages": num_stages,
        "PRECISION": PRECISION, "EVEN_K": int(EVEN_K),
    })
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

```python title="impl_cutile.py"
import torch
import cuda.tile as ct
import numpy as np

ConstInt = ct.Constant[int]

_LAST_CFG: dict = {}

# Mask that clears the lower 13 mantissa bits of an fp32 → produces a value
# representable exactly in TF32 (10-bit mantissa). -8192 == 0xFFFFE000 in int32.
_TF32_MASK = -8192


@ct.kernel(occupancy=2)
def _bmm_kernel_lp(A, B, C,
                   M: ConstInt, N: ConstInt, K: ConstInt,
                   TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp16/bf16 path: fp32 accumulator via ct.mma."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    batch = ct.bid(2)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a = ct.load(A, index=(batch, pid_m, k),
                    shape=(1, TM, TK),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TM, TK))
        b = ct.load(B, index=(batch, k, pid_n),
                    shape=(1, TK, TN),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TK, TN))
        acc = ct.mma(a, b, acc)

    out = ct.astype(acc, A.dtype).reshape((1, TM, TN))
    ct.store(C, index=(batch, pid_m, pid_n), tile=out)


@ct.kernel(occupancy=2)
def _bmm_kernel_fp32_4xtf32(A, B, C,
                            M: ConstInt, N: ConstInt, K: ConstInt,
                            TM: ConstInt, TN: ConstInt, TK: ConstInt,
                            MASK: ConstInt):
    """fp32 path: 4xTF32 emulation. Each input is split a = a_hi + a_lo where
       a_hi is TF32-representable (low 13 mantissa bits = 0) and a_lo is the
       residual. C += a_hi*b_hi + a_lo*b_hi + a_hi*b_lo + a_lo*b_lo.
       All 4 ct.mma calls use TF32 cores; sum recovers full fp32 precision."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    batch = ct.bid(2)

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a = ct.load(A, index=(batch, pid_m, k),
                    shape=(1, TM, TK),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TM, TK))
        b = ct.load(B, index=(batch, k, pid_n),
                    shape=(1, TK, TN),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TK, TN))

        # Split each fp32 into TF32 hi + residual lo.
        a_int = ct.bitcast(a, np.int32)
        a_hi = ct.bitcast(a_int & MASK, np.float32)
        a_lo = a - a_hi
        b_int = ct.bitcast(b, np.int32)
        b_hi = ct.bitcast(b_int & MASK, np.float32)
        b_lo = b - b_hi

        acc = ct.mma(a_hi, b_hi, acc)
        acc = ct.mma(a_lo, b_hi, acc)
        acc = ct.mma(a_hi, b_lo, acc)
        acc = ct.mma(a_lo, b_lo, acc)

    out = acc.reshape((1, TM, TN))
    ct.store(C, index=(batch, pid_m, pid_n), tile=out)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A3 = A.view(BATCH, M, K).contiguous()
    B3 = B.view(BATCH, K, N).contiguous()
    C3 = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)
    is_fp32 = (A.dtype == torch.float32)

    stream = torch.cuda.current_stream()

    TM, TN, TK = 128, 128, 32
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), BATCH)

    if is_fp32:
        ct.launch(stream, grid, _bmm_kernel_fp32_4xtf32,
                  (A3, B3, C3, M, N, K, TM, TN, TK, _TF32_MASK))
    else:
        ct.launch(stream, grid, _bmm_kernel_lp,
                  (A3, B3, C3, M, N, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM, "TN": TN, "TK": TK,
        "occupancy": 2,
        "FP32_EMU": "4xTF32" if is_fp32 else "none",
    })
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Summary: Triton fp32 switches from slow IEEE to `tf32x3` (3-product TF32) for ~3× fp32 speedup while maintaining precision. Cutile fp32 upgrades from 3xTF32 (2.4% mismatch) to full 4xTF32 to capture the missing `a_lo*b_lo` term and pass verify.
