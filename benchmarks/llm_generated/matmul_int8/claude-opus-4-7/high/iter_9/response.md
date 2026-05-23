Looking at the trajectory, iter 5 was best (9.7%) with `BLOCK_M=128, BLOCK_N=256, BLOCK_KB=64, stages=4`. Iter 6 attempted a fused single-`tl.dot` approach with `BLOCK_K_DOT=256` and regressed (likely because 256 exceeds int8's max recommended K-tile=128).

**New strategy for triton**: fused single `tl.dot` with `BLOCK_K_DOT = 4*32 = 128` (sweet spot for int8 tensor cores). Build the full `(4*BLOCK_KB, BLOCK_N)` unpacked-B tile via 3D broadcast+reshape, and load A as one wide `(BLOCK_M, 4*BLOCK_KB)` tile. One big int8 IMMA per outer iter instead of four small ones.

**For cutile**: keep iter 8's best config (4.9%) since multiple variants tried — small config tweak unlikely to move much.

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

    BLOCK_K_DOT: tl.constexpr = 4 * BLOCK_KB

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_KB)
    offs_i = tl.arange(0, 4)
    offs_j = tl.arange(0, BLOCK_K_DOT)
    i_per_j = offs_j // BLOCK_KB
    kb_per_j = offs_j % BLOCK_KB

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in range(0, K_b, BLOCK_KB):
        kb_idx = kb_start + offs_kb
        kb_mask = kb_idx < K_b

        # Load b_packed once: (BLOCK_KB, BLOCK_N) uint8
        b_packed = tl.load(
            b_ptr + kb_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=kb_mask[:, None] & n_mask,
            other=0,
        )
        b_packed_i32 = b_packed.to(tl.int32)

        # Build (4, BLOCK_KB, BLOCK_N) unpacked, then reshape to (4*BLOCK_KB, BLOCK_N)
        shifts = 2 * offs_i  # (4,)
        b_3d = ((b_packed_i32[None, :, :] >> shifts[:, None, None]) & 3) - 1
        b_unpacked = tl.reshape(b_3d.to(tl.int8), (BLOCK_K_DOT, BLOCK_N))

        # Load A as (BLOCK_M, 4*BLOCK_KB) where position j corresponds to
        # k = (j // BLOCK_KB) * K_b + (kb_start + j % BLOCK_KB)
        a_kb_global = kb_start + kb_per_j
        a_k = i_per_j * K_b + a_kb_global
        a_k_mask = a_kb_global < K_b
        a_tile = tl.load(
            a_ptr + offs_m[:, None] * stride_am + a_k[None, :] * stride_ak,
            mask=m_mask & a_k_mask[None, :],
            other=0,
        )

        # One large int8 IMMA per outer iteration
        acc = tl.dot(a_tile, b_unpacked, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_KB = 32
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
        "BLOCK_K_DOT": 4 * BLOCK_KB,
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
    TKB = 128
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
