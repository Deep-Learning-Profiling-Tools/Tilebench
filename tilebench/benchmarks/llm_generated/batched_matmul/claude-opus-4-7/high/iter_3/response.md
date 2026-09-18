Looking at the trajectory:
- Triton iter 2 broke fp32 verify by switching to TF32x3. Revert to iter 1's IEEE fp32 path and try **BLOCK_K=64** for fewer K iterations and better compute/load ratio.
- cuTile iter 1/2 both failed fp32 verify. The TF32x3 attempt via `ct.astype(tfloat32)` was buggy (83% mismatch). Try a **Veltkamp split** using float arithmetic only to manually split fp32 into TF32-precision hi+lo halves, then 3-pass MMA.

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
                PRECISION: tl.constexpr):
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
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        if PRECISION == 0:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
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

    # IEEE fp32 to match torch's fp32 matmul precision.
    PRECISION = 0 if A.dtype == torch.float32 else 1

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), BATCH)
    _bmm_kernel[grid](
        A3, B3, C3,
        M, N, K,
        A3.stride(0), A3.stride(1), A3.stride(2),
        B3.stride(0), B3.stride(1), B3.stride(2),
        C3.stride(0), C3.stride(1), C3.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M, PRECISION=PRECISION,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K,
        "GROUP_M": GROUP_M, "num_warps": num_warps, "num_stages": num_stages,
        "PRECISION": PRECISION,
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
ConstBool = ct.Constant[bool]

_LAST_CFG: dict = {}


@ct.kernel(occupancy=2)
def _bmm_kernel_lp(A, B, C,
                   M: ConstInt, N: ConstInt, K: ConstInt,
                   TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp16/bf16 path: single MMA per K-tile."""
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


@ct.kernel(occupancy=1)
def _bmm_kernel_fp32(A, B, C,
                     M: ConstInt, N: ConstInt, K: ConstInt,
                     TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp32 path: TF32x3 emulation via Veltkamp split (a = a_hi + a_lo
    where a_hi has only ~10 mantissa bits, so the TF32 tensor cores see
    a_hi/b_hi/a_lo/b_lo losslessly). 3 MMAs per K-tile approximate IEEE
    fp32 precision."""
    pid_m = ct.bid(0)
    pid_n = ct.bid(1)
    batch = ct.bid(2)

    # Veltkamp split constant: C = 2^13 + 1 splits fp32 (23-bit mantissa)
    # into hi (10-bit, matching TF32) + lo residual.
    SPLIT_C = 8193.0

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)
    num_k = ct.cdiv(K, TK)
    for k in range(0, num_k):
        a = ct.load(A, index=(batch, pid_m, k),
                    shape=(1, TM, TK),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TM, TK))
        b = ct.load(B, index=(batch, k, pid_n),
                    shape=(1, TK, TN),
                    padding_mode=ct.PaddingMode.ZERO).reshape((TK, TN))

        # Split a into (a_hi, a_lo); both fp32 but a_hi has TF32 precision.
        ca = SPLIT_C * a
        a_hi = ca - (ca - a)
        a_lo = a - a_hi

        cb = SPLIT_C * b
        b_hi = cb - (cb - b)
        b_lo = b - b_hi

        # 3 MMAs (drops a_lo * b_lo term which is ~2^-20 relative).
        acc = ct.mma(a_hi, b_hi, acc)
        acc = ct.mma(a_lo, b_hi, acc)
        acc = ct.mma(a_hi, b_lo, acc)

    out = acc.reshape((1, TM, TN))
    ct.store(C, index=(batch, pid_m, pid_n), tile=out)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A3 = A.view(BATCH, M, K).contiguous()
    B3 = B.view(BATCH, K, N).contiguous()
    C3 = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)

    TM = 128
    TN = 128
    TK = 32
    is_fp32 = (A.dtype == torch.float32)

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), BATCH)
    if is_fp32:
        ct.launch(stream, grid, _bmm_kernel_fp32,
                  (A3, B3, C3, M, N, K, TM, TN, TK))
    else:
        ct.launch(stream, grid, _bmm_kernel_lp,
                  (A3, B3, C3, M, N, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TM": TM, "TN": TN, "TK": TK,
                      "occupancy": 1 if is_fp32 else 2,
                      "IS_FP32": is_fp32})
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: Triton reverts to verified iter-1 (IEEE fp32) and bumps `BLOCK_K` from 32→64 to halve K-iteration overhead. cuTile keeps simple MMA for fp16/bf16 but adds a Veltkamp-split TF32x3 path for fp32 (decompose `a = a_hi + a_lo` via pure float arithmetic so each half has only TF32 precision, then 3 MMAs reconstruct ~IEEE accuracy through the TF32 cores).
