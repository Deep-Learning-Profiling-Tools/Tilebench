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

    a_batch_base = batch_id * M * K
    b_batch_base = batch_id * K * N
    c_batch_base = batch_id * M * N

    a_block = tl.make_block_ptr(
        base=A_ptr + a_batch_base,
        shape=(M, K),
        strides=(K, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=B_ptr + b_batch_base,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(
            a_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        b = tl.load(
            b_block,
            boundary_check=(0, 1),
            padding_option="zero",
        )
        acc = tl.dot(a, b, acc, input_precision="tf32x3")
        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    c_block = tl.make_block_ptr(
        base=C_ptr + c_batch_base,
        shape=(M, N),
        strides=(N, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(c_block, acc, boundary_check=(0, 1))


def run(A: torch.Tensor, B: torch.Tensor,
        BATCH: int, M: int, N: int, K: int, **kwargs):
    C = torch.empty((BATCH * M * N,), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 3

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
        "input_precision": "tf32x3",
        "load_path": "block_ptr",
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


@ct.kernel
def _batched_matmul_kernel(
    A,
    B,
    C,
    K: ConstInt,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    USE_TF32X3: ConstBool,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    bid_b = ct.bid(2)

    num_k_tiles = ct.cdiv(K, TILE_K)

    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)

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

        if USE_TF32X3:
            a_f = ct.astype(a_tile, np.float32)
            b_f = ct.astype(b_tile, np.float32)

            a_hi = ct.astype(a_f, ct.tfloat32)
            b_hi = ct.astype(b_f, ct.tfloat32)

            a_hi_f = ct.astype(a_hi, np.float32)
            b_hi_f = ct.astype(b_hi, np.float32)

            a_lo = ct.astype(a_f - a_hi_f, ct.tfloat32)
            b_lo = ct.astype(b_f - b_hi_f, ct.tfloat32)

            acc = ct.mma(a_hi, b_hi, acc)
            acc = ct.mma(a_hi, b_lo, acc)
            acc = ct.mma(a_lo, b_hi, acc)
            acc = ct.mma(a_lo, b_lo, acc)
        else:
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
    USE_TF32X3 = (A.dtype == torch.float32)

    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), BATCH)
    stream = torch.cuda.current_stream()
    kernel = _batched_matmul_kernel.with_hints(occupancy=occupancy)

    ct.launch(
        stream,
        grid,
        kernel,
        (A_3d, B_3d, C_3d, K, TILE_M, TILE_N, TILE_K, USE_TF32X3),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TILE_M": TILE_M,
        "TILE_N": TILE_N,
        "TILE_K": TILE_K,
        "occupancy": occupancy,
        "fp32_precision": "tf32x3_4term",
    })
    return C_3d.view(-1)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
