Looking at the trajectory:
- Triton best is iter 3 (9.7%): BLOCK_M=128, BLOCK_N=256, BLOCK_KB=64, num_stages=3. Iter 4 regressed by increasing BLOCK_KB.
- Cutile best is iter 2 (4.5%): TM=128, TN=128, TKB=64. Wider TN regressed.

I'll revert to the best configs and try **deeper pipelining** (num_stages=4) for Triton given K=20480 has 320 K-block iterations. For cuTile, try a larger K tile (TKB=128) at TN=128 to amortize B loads better.

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

    a_base = a_ptr + offs_m[:, None] * stride_am
    b_base = b_ptr + offs_n[None, :] * stride_bn

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in range(0, K_b, BLOCK_KB):
        kb_idx = kb_start + offs_kb
        kb_mask = kb_idx < K_b
        b_packed = tl.load(
            b_base + kb_idx[:, None] * stride_bk,
            mask=kb_mask[:, None] & n_mask,
            other=0,
        )
        for i in tl.static_range(4):
            shifted = (b_packed >> (2 * i)) & 3
            b_val = shifted.to(tl.int8) - 1
            k_offset_a = i * K_b + kb_idx
            a_val = tl.load(
                a_base + k_offset_a[None, :] * stride_ak,
                mask=m_mask & kb_mask[None, :],
                other=0,
            )
            acc = tl.dot(a_val, b_val, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = m_mask & n_mask
    tl.store(c_ptrs, acc, mask=c_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_KB = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 4

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

Approach: Triton — revert to iter-3 best config (BLOCK_M=128, BLOCK_N=256, BLOCK_KB=64) but bump `num_stages=4` to exploit deeper pipelining over the long K=20480 dimension (320 K-iterations gives plenty of headroom). cuTile — revert to iter-2 best (TM=128, TN=128) but double TKB to 128 so each B-packed load amortizes more A loads per outer K iteration.
