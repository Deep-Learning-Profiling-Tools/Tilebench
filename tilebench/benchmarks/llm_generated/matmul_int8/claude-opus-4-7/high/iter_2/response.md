Looking at iter 1 (6.6% roofline), the main inefficiency is the 4 separate `tl.dot` calls per K-block, each with small inner-K=BLOCK_KB. The key optimization: **concatenate the 4 unpacked B fields into a single big dot** with effective K = 4×BLOCK_KB. I do this with 3D loads/tiles and a final reshape, which keeps a contiguous innermost axis for coalescing. For cuTile, I fix the iter-1 compile failure by doing the unpack in int32 (cleaner integer arithmetic) and only casting to int8 right before MMA.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_int8_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K_b,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KB: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
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
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_KB)
    offs_i = tl.arange(0, 4)
    shifts = offs_i * 2  # [0, 2, 4, 6]

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in range(0, K_b, BLOCK_KB):
        kb_idx = kb_start + offs_kb
        kb_mask = kb_idx < K_b

        # ----- Load B packed (BLOCK_KB, BLOCK_N) and unpack into 4 fields -----
        b_ptrs = b_ptr + kb_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_packed = tl.load(b_ptrs, mask=kb_mask[:, None] & n_mask[None, :], other=0)
        b_packed_i32 = b_packed.to(tl.int32)
        # (4, BLOCK_KB, BLOCK_N): one int8 plane per field
        b_unpacked = ((b_packed_i32[None, :, :] >> shifts[:, None, None]) & 3) - 1
        b_unpacked = b_unpacked.to(tl.int8)
        # Flatten field axis into K: result[i*BLOCK_KB + kb, n] = field_i(b_packed[kb, n])
        b_full = tl.reshape(b_unpacked, (4 * BLOCK_KB, BLOCK_N))

        # ----- Load A as (BLOCK_M, 4, BLOCK_KB), where field i sits at K-offset i*K_b -----
        a_ptrs = (a_ptr
                  + offs_m[:, None, None] * stride_am
                  + offs_i[None, :, None] * K_b * stride_ak
                  + kb_idx[None, None, :] * stride_ak)
        a_mask_3d = m_mask[:, None, None] & kb_mask[None, None, :]
        a_3d = tl.load(a_ptrs, mask=a_mask_3d, other=0)
        # Flatten field axis into K to match b_full's layout
        a_full = tl.reshape(a_3d, (BLOCK_M, 4 * BLOCK_KB))

        acc = tl.dot(a_full, b_full, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(c_ptrs, acc, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_KB = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 3

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _matmul_int8_kernel[grid](
        a, b, output,
        M, N, K_b,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_KB=BLOCK_KB,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_KB": BLOCK_KB,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_warps": num_warps,
        "num_stages": num_stages,
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
def _matmul_int8_kernel(
    a, b, out,
    K_b_tiles: ConstInt,
    TM: ConstInt, TN: ConstInt, TKB: ConstInt,
):
    bm = ct.bid(0)
    bn = ct.bid(1)

    acc = ct.full((TM, TN), 0, dtype=np.int32)

    for kb_block in range(0, K_b_tiles):
        b_packed = ct.load(
            b, index=(kb_block, bn), shape=(TKB, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_packed_i32 = ct.astype(b_packed, np.int32)
        for i in range(4):
            b_i32 = ((b_packed_i32 >> (2 * i)) & 3) - 1
            b_int8 = ct.astype(b_i32, np.int8)
            a_tile = ct.load(
                a, index=(bm, i * K_b_tiles + kb_block),
                shape=(TM, TKB),
                padding_mode=ct.PaddingMode.ZERO,
            )
            acc = ct.mma(a_tile, b_int8, acc)

    ct.store(out, index=(bm, bn), tile=acc)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    TM = 128
    TN = 128
    TKB = 64
    occupancy = 2

    K_b_tiles = (K_b + TKB - 1) // TKB

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(M, TM), ct.cdiv(N, TN), 1)
    ct.launch(stream, grid, _matmul_int8_kernel,
              (a, b, output, K_b_tiles, TM, TN, TKB))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM, "TN": TN, "TKB": TKB,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
