```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _batched_matmul_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    BATCH,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    USE_FP32_DECOMP: tl.constexpr,
):
    tile_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_in_group = tile_id - group_id * num_pid_in_group
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    tl.assume(batch_id >= 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_batch_base = batch_id * M * K
    b_batch_base = batch_id * K * N
    c_batch_base = batch_id * M * N

    a_ptrs = A_ptr + a_batch_base + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + b_batch_base + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_tile in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_tile * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)

        if USE_FP32_DECOMP:
            a_f = tl.load(
                a_ptrs,
                mask=a_mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            b_f = tl.load(
                b_ptrs,
                mask=b_mask,
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)

            a0 = a_f.to(tl.float16)
            b0 = b_f.to(tl.float16)
            a1 = (a_f - a0.to(tl.float32)).to(tl.float16)
            b1 = (b_f - b0.to(tl.float32)).to(tl.float16)

            acc = tl.dot(a0, b0, acc)
            acc = tl.dot(a0, b1, acc)
            acc = tl.dot(a1, b0, acc)
            acc = tl.dot(a1, b1, acc)
        else:
            a = tl.load(
                a_ptrs,
                mask=a_mask,
                other=0.0,
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_ptrs,
                mask=b_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            acc = tl.dot(a, b, acc, input_precision="ieee")

        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    c_ptrs = C_ptr + c_batch_base + offs_m[:, None] * N + offs_n[None, :]
    c_mask = (batch_id < BATCH) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    C = torch.empty((BATCH * M * N,), device=A.device, dtype=A.dtype)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_warps = 4
    num_stages = 3
    USE_FP32_DECOMP = (A.dtype == torch.float32)

    grid = (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
        BATCH,
    )

    _batched_matmul_kernel[grid](
        A,
        B,
        C,
        BATCH,
        M,
        N,
        K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        USE_FP32_DECOMP=USE_FP32_DECOMP,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "GROUP_SIZE_M": GROUP_SIZE_M,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "input_precision": "ieee",
        "fp32_mode": "fp16_2chunk_4term",
        "USE_FP32_DECOMP": USE_FP32_DECOMP,
    })
    return C


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
def _batched_matmul_kernel(
    A,
    B,
    C,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    USE_FP32_PRECISE: ConstBool,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    bid_b = ct.bid(2)

    num_k_tiles = ct.cdiv(K, TILE_K)
    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)

    if USE_FP32_PRECISE:
        for k_tile in range(0, num_k_tiles):
            a_tile_3d = ct.load(
                A,
                index=(bid_b, bid_m, k_tile),
                shape=(1, TILE_M, TILE_K),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_tile_3d = ct.load(
                B,
                index=(bid_b, k_tile, bid_n),
                shape=(1, TILE_K, TILE_N),
                padding_mode=ct.PaddingMode.ZERO,
            )

            a_f = ct.astype(ct.reshape(a_tile_3d, (TILE_M, TILE_K)), np.float32)
            b_f = ct.astype(ct.reshape(b_tile_3d, (TILE_K, TILE_N)), np.float32)

            a0 = ct.astype(a_f, ct.tfloat32)
            b0 = ct.astype(b_f, ct.tfloat32)
            a0_f = ct.astype(a0, np.float32)
            b0_f = ct.astype(b0, np.float32)

            ar1 = a_f - a0_f
            br1 = b_f - b0_f
            a1 = ct.astype(ar1, ct.tfloat32)
            b1 = ct.astype(br1, ct.tfloat32)
            a1_f = ct.astype(a1, np.float32)
            b1_f = ct.astype(b1, np.float32)

            ar2 = ar1 - a1_f
            br2 = br1 - b1_f
            a2 = ct.astype(ar2, ct.tfloat32)
            b2 = ct.astype(br2, ct.tfloat32)
            a2_f = ct.astype(a2, np.float32)
            b2_f = ct.astype(b2, np.float32)

            ar3 = ar2 - a2_f
            br3 = br2 - b2_f
            a3 = ct.astype(ar3, ct.tfloat32)
            b3 = ct.astype(br3, ct.tfloat32)

            acc = ct.mma(a0, b0, acc)

            acc = ct.mma(a0, b1, acc)
            acc = ct.mma(a1, b0, acc)

            acc = ct.mma(a0, b2, acc)
            acc = ct.mma(a1, b1, acc)
            acc = ct.mma(a2, b0, acc)

            acc = ct.mma(a0, b3, acc)
            acc = ct.mma(a1, b2, acc)
            acc = ct.mma(a2, b1, acc)
            acc = ct.mma(a3, b0, acc)

            acc = ct.mma(a1, b3, acc)
            acc = ct.mma(a2, b2, acc)
            acc = ct.mma(a3, b1, acc)

            acc = ct.mma(a2, b3, acc)
            acc = ct.mma(a3, b2, acc)

            acc = ct.mma(a3, b3, acc)
    else:
        for k_tile in range(0, num_k_tiles):
            a_tile_3d = ct.load(
                A,
                index=(bid_b, bid_m, k_tile),
                shape=(1, TILE_M, TILE_K),
                padding_mode=ct.PaddingMode.ZERO,
            )
            b_tile_3d = ct.load(
                B,
                index=(bid_b, k_tile, bid_n),
                shape=(1, TILE_K, TILE_N),
                padding_mode=ct.PaddingMode.ZERO,
            )

            a_tile = ct.reshape(a_tile_3d, (TILE_M, TILE_K))
            b_tile = ct.reshape(b_tile_3d, (TILE_K, TILE_N))
            acc = ct.mma(a_tile, b_tile, acc)

    out_tile = ct.astype(acc, A.dtype)
    out_tile_3d = ct.reshape(out_tile, (1, TILE_M, TILE_N))
    ct.store(C, index=(bid_b, bid_m, bid_n), tile=out_tile_3d)


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    A_3d = A.view(BATCH, M, K)
    B_3d = B.view(BATCH, K, N)
    C_3d = torch.empty((BATCH, M, N), device=A.device, dtype=A.dtype)

    TILE_M = 128
    TILE_N = 128
    TILE_K = 32
    occupancy = 2
    USE_FP32_PRECISE = (A.dtype == torch.float32)

    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), BATCH)
    stream = torch.cuda.current_stream()

    ct.launch(
        stream,
        grid,
        _batched_matmul_kernel,
        (A_3d, B_3d, C_3d, K, TILE_M, TILE_N, TILE_K, USE_FP32_PRECISE),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE_M": TILE_M,
        "TILE_N": TILE_N,
        "TILE_K": TILE_K,
        "occupancy": occupancy,
        "fp32_precision": "tf32_4chunk_16term",
        "USE_FP32_PRECISE": USE_FP32_PRECISE,
    })
    return C_3d.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
