Looking at the trajectory, iter 3 and 5 both hit 9.7% for triton with the multi-dot approach. The plateau suggests we need an algorithmic change rather than tuning. The current kernel issues 4 separate `tl.dot(BLOCK_M, BLOCK_KB, BLOCK_N)` per inner step. Combining them into a **single big dot** with K-dim = `4*BLOCK_KB` should let the compiler schedule one larger MMA chain, improving tensor-core utilization. I apply the same idea to cuTile.

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

    BLOCK_K4: tl.constexpr = 4 * BLOCK_KB

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kb = tl.arange(0, BLOCK_KB)
    offs_kk = tl.arange(0, BLOCK_K4)
    i_idx = offs_kk // BLOCK_KB        # 0..3
    kb_local = offs_kk % BLOCK_KB      # 0..BLOCK_KB-1

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for kb_start in range(0, K_b, BLOCK_KB):
        kb_idx = kb_start + offs_kb
        kb_mask = kb_idx < K_b

        # Load packed B: (BLOCK_KB, BLOCK_N) uint8
        b_packed = tl.load(
            b_ptr + kb_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=kb_mask[:, None] & n_mask,
            other=0,
        )
        b_packed_i32 = b_packed.to(tl.int32)

        # Load A_big: (BLOCK_M, 4*BLOCK_KB), where column k_meta=(i,kb_local)
        # references K-index = i*K_b + kb_start + kb_local.
        a_k_offs = i_idx * K_b + kb_start + kb_local
        a_k_mask = (kb_start + kb_local) < K_b
        a_big = tl.load(
            a_ptr + offs_m[:, None] * stride_am + a_k_offs[None, :] * stride_ak,
            mask=m_mask & a_k_mask[None, :],
            other=0,
        )

        # Build B_big: (4*BLOCK_KB, BLOCK_N) int8 via 3D broadcast + reshape
        shifts = (tl.arange(0, 4) * 2)[:, None, None]       # (4, 1, 1)
        b_3d = b_packed_i32[None, :, :]                      # (1, BLOCK_KB, BLOCK_N)
        b_unp = ((b_3d >> shifts) & 3) - 1                   # (4, BLOCK_KB, BLOCK_N)
        b_big = tl.reshape(b_unp, (BLOCK_K4, BLOCK_N)).to(tl.int8)

        acc = tl.dot(a_big, b_big, acc, out_dtype=tl.int32)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=m_mask & n_mask)


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    K_b, N = b.shape
    output = torch.empty((M, N), dtype=torch.int32, device=a.device)

    BLOCK_M = 128
    BLOCK_N = 256
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

        # Unpack into 4 (TKB, TN) int8 tiles
        b0 = ct.astype(((b_packed_i32 >> 0) & 3) - 1, np.int8)
        b1 = ct.astype(((b_packed_i32 >> 2) & 3) - 1, np.int8)
        b2 = ct.astype(((b_packed_i32 >> 4) & 3) - 1, np.int8)
        b3 = ct.astype(((b_packed_i32 >> 6) & 3) - 1, np.int8)
        # Concat along axis 0 → (4*TKB, TN)
        b_big = ct.cat(
            (ct.cat((b0, b1), axis=0), ct.cat((b2, b3), axis=0)),
            axis=0,
        )

        # Load 4 (TM, TKB) chunks of A and concat along axis 1 → (TM, 4*TKB)
        a0 = ct.load(a, index=(bm, 0 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a1 = ct.load(a, index=(bm, 1 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a2 = ct.load(a, index=(bm, 2 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a3 = ct.load(a, index=(bm, 3 * K_b_tiles + kb_block),
                     shape=(TM, TKB), padding_mode=ct.PaddingMode.ZERO)
        a_big = ct.cat(
            (ct.cat((a0, a1), axis=1), ct.cat((a2, a3), axis=1)),
            axis=1,
        )

        acc = ct.mma(a_big, b_big, acc)

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
        "TK_DOT": 4 * TKB,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: collapse the 4 inner dots into a single large `tl.dot` / `ct.mma` of K-dim `4*BLOCK_KB`, by building A_big = `(BLOCK_M, 4*BLOCK_KB)` via strided loads and B_big = `(4*BLOCK_KB, BLOCK_N)` via per-shift unpack + concat. This gives the compiler one big MMA chain to schedule, improving tensor-core throughput vs four smaller ones with identical total work.
