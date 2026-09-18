```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _zero_i32_kernel(c_ptr, total, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    z = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    tl.store(c_ptr + offs, z, mask=mask)


@triton.jit
def _matmul_int8_packed_splitk_kernel(
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
    SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    split_id = tl.program_id(1)

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
    tl.assume(split_id >= 0)
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

    kb_per_split = Kb // SPLIT_K
    kb_begin = split_id * kb_per_split
    kb_end = kb_begin + kb_per_split

    for kb_start in tl.range(kb_begin, kb_end, BLOCK_K, num_stages=LOOP_STAGES):
        kb_offsets = kb_start + offs_k

        b_ptrs = b_ptr + kb_offsets[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_bytes = tl.load(b_ptrs, eviction_policy="evict_last")

        for field in tl.static_range(0, 4):
            a_cols = field * Kb + kb_offsets
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + a_cols[None, :] * stride_ak
            a_tile = tl.load(a_ptrs, eviction_policy="evict_first").to(tl.int8)

            b_bits = (b_bytes.to(tl.int32) >> (2 * field)) & 3
            b_vals = (b_bits - 1).to(tl.int8)

            acc = tl.dot(a_tile, b_vals, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.atomic_add(c_ptrs, acc, mask=c_mask, sem="relaxed")


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
    SPLIT_K = 4
    num_warps = 8
    num_stages = 4

    ZERO_BLOCK_SIZE = 1024
    zero_num_warps = 4
    zero_num_stages = 2

    total = output.numel()
    zero_grid = (triton.cdiv(total, ZERO_BLOCK_SIZE),)
    _zero_i32_kernel[zero_grid](
        output,
        total,
        BLOCK_SIZE=ZERO_BLOCK_SIZE,
        num_warps=zero_num_warps,
        num_stages=zero_num_stages,
    )

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), SPLIT_K)

    _matmul_int8_packed_splitk_kernel[grid](
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
        SPLIT_K=SPLIT_K,
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
            "SPLIT_K": SPLIT_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOAD_MASKS": 0,
            "ATOMIC_SPLITK": 1,
            "ZERO_BLOCK_SIZE": ZERO_BLOCK_SIZE,
            "zero_num_warps": zero_num_warps,
            "zero_num_stages": zero_num_stages,
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


@ct.kernel(occupancy=4)
def _zero_output_kernel(
    output,
    TM: ConstInt,
    TN: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    z = ct.full((TM, TN), 0, dtype=np.int32)
    ct.store(output, index=(bid_m, bid_n), tile=z)


@ct.kernel(occupancy=2)
def _matmul_int8_packed_splitk_kernel(
    a,
    b,
    output,
    KB: ConstInt,
    TM: ConstInt,
    TN: ConstInt,
    TK: ConstInt,
    SPLIT_K: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    split_id = ct.bid(2)

    acc = ct.full((TM, TN), 0, dtype=np.int32)

    k_tiles_per_split = KB // (SPLIT_K * TK)
    k_tile_base = split_id * k_tiles_per_split

    for j in range(0, k_tiles_per_split):
        kj = k_tile_base + j

        b_tile = ct.load(
            b,
            index=(kj, bid_n),
            shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )

        for field in range(0, 4):
            a_field = a.slice(1, field * KB, (field + 1) * KB)
            a_tile = ct.load(
                a_field,
                index=(bid_m, kj),
                shape=(TM, TK),
                padding_mode=ct.PaddingMode.ZERO,
            )

            shifted = ct.bitwise_rshift(ct.astype(b_tile, np.int32), 2 * field)
            bits = ct.bitwise_and(shifted, 3)
            b_vals = ct.astype(bits - 1, np.int8)

            acc = ct.mma(a_tile, b_vals, acc)

    offs_m = bid_m * TM + ct.arange(TM, dtype=np.int32)[:, None]
    offs_n = bid_n * TN + ct.arange(TN, dtype=np.int32)[None, :]

    ct.atomic_add(
        output,
        (offs_m, offs_n),
        acc,
        memory_order=ct.MemoryOrder.RELAXED,
        memory_scope=ct.MemoryScope.DEVICE,
    )


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    KB = b.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    TM = 128
    TN = 128
    TK = 128
    SPLIT_K = 2
    occupancy = 2
    zero_occupancy = 4

    tile_grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, tile_grid, _zero_output_kernel, (output, TM, TN))

    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), SPLIT_K)
    ct.launch(
        stream,
        grid,
        _matmul_int8_packed_splitk_kernel,
        (a, b, output, KB, TM, TN, TK, SPLIT_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TM": TM,
            "TN": TN,
            "TK": TK,
            "SPLIT_K": SPLIT_K,
            "occupancy": occupancy,
            "zero_occupancy": zero_occupancy,
            "ATOMIC_SPLITK": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
