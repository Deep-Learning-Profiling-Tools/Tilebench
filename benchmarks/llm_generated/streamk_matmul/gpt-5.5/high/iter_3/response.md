```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr,
                   M, N, K,
                   stride_am, stride_ak,
                   stride_bk, stride_bn,
                   BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr,
                   BLOCK_K: tl.constexpr,
                   GROUP_SIZE_M: tl.constexpr,
                   PIPE_STAGES: tl.constexpr):
    pid = tl.program_id(0)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume((M % BLOCK_M) == 0)
    tl.assume((N % BLOCK_N) == 0)
    tl.assume((K % BLOCK_K) == 0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_in_group = pid - group_id * num_pid_in_group

    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_m = tl.max_contiguous(tl.multiple_of(offs_m, BLOCK_M), BLOCK_M)
    offs_n = tl.max_contiguous(tl.multiple_of(offs_n, BLOCK_N), BLOCK_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
        a = tl.load(a_ptrs, eviction_policy="evict_first")
        b = tl.load(b_ptrs, eviction_policy="evict_last")
        acc = tl.dot(a, b, acc, input_precision="tf32")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc)


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 2
    PIPE_STAGES = 2

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _matmul_kernel[grid](
        a, b, output,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        PIPE_STAGES=PIPE_STAGES,
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
        "PIPE_STAGES": PIPE_STAGES,
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


@ct.kernel(occupancy=1)
def _matmul_kernel(a, b, out,
                   K_TILES: ConstInt,
                   TM: ConstInt, TN: ConstInt, TK: ConstInt):
    pid = ct.bid(0)

    num_pid_n = ct.num_tiles(out, 1, (TM, TN))
    pid_m = pid // num_pid_n
    pid_n = pid - pid_m * num_pid_n

    acc = ct.full((TM, TN), 0.0, dtype=np.float32)

    for kk in range(0, K_TILES):
        a_tile = ct.load(
            a,
            index=(pid_m, kk),
            shape=(TM, TK),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            b,
            index=(kk, pid_n),
            shape=(TK, TN),
            padding_mode=ct.PaddingMode.ZERO,
        )

        if a.dtype == ct.float32:
            a_tile = ct.astype(a_tile, ct.tfloat32)
            b_tile = ct.astype(b_tile, ct.tfloat32)

        acc = ct.mma(a_tile, b_tile, acc)

    ct.store(out, index=(pid_m, pid_n), tile=ct.astype(acc, out.dtype))


def run(a: torch.Tensor, b: torch.Tensor, **kwargs):
    m = a.shape[0]
    k = a.shape[1]
    n = b.shape[1]

    output = torch.empty((m, n), device=a.device, dtype=a.dtype)

    TM = 128
    TN = 128
    TK = 64
    K_TILES = (k + TK - 1) // TK
    occupancy = 1

    grid_m = (m + TM - 1) // TM
    grid_n = (n + TN - 1) // TN
    grid = (grid_m * grid_n, 1, 1)

    stream = torch.cuda.current_stream()
    ct.launch(stream, grid, _matmul_kernel, (a, b, output, K_TILES, TM, TN, TK))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "TM": TM,
        "TN": TN,
        "TK": TK,
        "K_TILES": K_TILES,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
