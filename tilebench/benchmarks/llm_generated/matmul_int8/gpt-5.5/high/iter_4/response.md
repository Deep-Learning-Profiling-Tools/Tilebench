```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_int8_packed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    Kb,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ak == 1)
    tl.assume(stride_bn == 1)
    tl.assume(stride_cn == 1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)
    offs_k = tl.max_contiguous(tl.multiple_of(offs_k, BLOCK_K), BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in tl.range(0, Kb, BLOCK_K, num_stages=LOOP_STAGES):
        kb_offsets = kb_start + offs_k

        b_ptrs = b_ptr + kb_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_bytes = tl.load(b_ptrs, eviction_policy="evict_last")

        for field in tl.static_range(0, 4):
            a_cols = field * Kb + kb_offsets
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + a_cols[None, :] * stride_ak
            a_tile = tl.load(a_ptrs, eviction_policy="evict_first").to(tl.int8)

            b_bits = (b_bytes >> (2 * field)) & 3
            b_vals = (b_bits.to(tl.int8) - 1).to(tl.int8)

            acc = tl.dot(a_tile, b_vals, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    Kb = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    LOOP_STAGES = 4
    num_warps = 8
    num_stages = 4

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _matmul_int8_packed_kernel[grid](
        a,
        b,
        output,
        M,
        N,
        Kb,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        LOOP_STAGES=LOOP_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "BLOCK_K": BLOCK_K,
            "GROUP_SIZE_M": GROUP_SIZE_M,
            "LOOP_STAGES": LOOP_STAGES,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOAD_MASKS": 0,
            "U8_SHIFT_DECODE": 1,
        }
    )
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
def _matmul_int8_packed_kernel(
    a,
    b,
    output,
    KB: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.full((TM, TN), 0, dtype=np.int32)
    num_k_tiles = ct.cdiv(KB, TK)

    a0_view = a.slice(1, 0, KB)
    a1_view = a.slice(1, KB, 2 * KB)
    a2_view = a.slice(1, 2 * KB, 3 * KB)
    a3_view = a.slice(1, 3 * KB, 4 * KB)

    for j in range(0, num_k_tiles):
        b_tile = ct.load(
            b,
            index=(j, bid_n),
            shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )

        b0_bits = ct.bitwise_and(b_tile, 3)
        b0 = ct.astype(ct.astype(b0_bits, np.int8) - 1, np.int8)
        a0 = ct.load(
            a0_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a0, b0, acc)

        b1_bits = ct.bitwise_and(ct.bitwise_rshift(b_tile, 2), 3)
        b1 = ct.astype(ct.astype(b1_bits, np.int8) - 1, np.int8)
        a1 = ct.load(
            a1_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a1, b1, acc)

        b2_bits = ct.bitwise_and(ct.bitwise_rshift(b_tile, 4), 3)
        b2 = ct.astype(ct.astype(b2_bits, np.int8) - 1, np.int8)
        a2 = ct.load(
            a2_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a2, b2, acc)

        b3_bits = ct.bitwise_and(ct.bitwise_rshift(b_tile, 6), 3)
        b3 = ct.astype(ct.astype(b3_bits, np.int8) - 1, np.int8)
        a3 = ct.load(
            a3_view,
            index=(bid_m, j),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        acc = ct.mma(a3, b3, acc)

    ct.store(output, index=(bid_m, bid_n), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    KB = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    TM = 128
    TN = 128
    TK = 128
    occupancy = 2

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, _matmul_int8_packed_kernel, (a, b, output, KB, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TM": TM,
            "TN": TN,
            "TK": TK,
            "occupancy": occupancy,
            "U8_SHIFT_DECODE": 1,
            "UNROLL_FIELDS": 4,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
