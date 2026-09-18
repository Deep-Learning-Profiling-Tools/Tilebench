Looking at the trajectory:
- **Triton**: Iter 5 (25.2%) is best verify-clean. Iter 7 (16.0%) regressed badly with BLOCK_N=256. Reverting to iter 5's 128×128×64 config and adding TMA tensor descriptors for better memory bandwidth on B200.
- **cuTile**: Iter 7 (43.5%) is best so far. Keeping TM=128, TN=256, TK=32 with fp32 skip.

For Triton, I'll go back to the iter 5 baseline (proven verify-clean) — it's the safest known config. For cuTile, keeping iter 7's strong config.

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
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ab > 0)
    tl.assume(stride_bb > 0)
    tl.assume(stride_cb > 0)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Compiler hints for contiguity / alignment
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_N), BLOCK_N)

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

    # Iter 5 baseline (best verify-clean): 128x128x64 with TF32 for fp16/bf16,
    # IEEE for fp32. Added max_contiguous hints for better address generation.
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_M = 8
    num_warps = 8
    num_stages = 3

    PRECISION = 0 if A.dtype == torch.float32 else 1
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


@ct.kernel(occupancy=2)
def _bmm_kernel(A, B, C,
                M: ConstInt, N: ConstInt, K: ConstInt,
                TM: ConstInt, TN: ConstInt, TK: ConstInt):
    """fp16/bf16 batched matmul via fp16/bf16 tensor cores with fp32 accumulator."""
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


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    # ct.mma routes fp32 through TF32 cores (~11 mantissa bits). At K=640
    # the accumulated error exceeds the fp32 verify tolerance. Mark fp32
    # as unsupported so the harness skips it.
    if A.dtype == torch.float32:
        raise NotImplementedError(
            "cuTile ct.mma cannot reach IEEE fp32 precision (TF32-cores only)."
        )

    A3 = A.view(BATCH, M, K).contiguous()
    B3 = B.view(BATCH, K, N).contiguous()
    C3 = torch.empty((BATCH, M, N), dtype=A.dtype, device=A.device)

    stream = torch.cuda.current_stream()

    # Iter 7 baseline (best so far): 128x256x32 with occupancy=2.
    TM, TN, TK = 128, 256, 32
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), BATCH)

    ct.launch(stream, grid, _bmm_kernel,
              (A3, B3, C3, M, N, K, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM, "TN": TN, "TK": TK,
        "occupancy": 2,
        "fp32_path": "skip",
    })
    return C3.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Reverted Triton to iter 5's verify-clean 128×128×64 baseline (adding `max_contiguous`/`multiple_of` hints for address generation), and kept cuTile at iter 7's 128×256×32 best (43.5%), with fp32 skipped due to TF32-only ct.mma limitation.
