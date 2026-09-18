```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
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
    NUM_STAGES: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)

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

    a_block = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in tl.range(0, K, BLOCK_K, num_stages=NUM_STAGES, warp_specialize=True):
        if EVEN_M and EVEN_N and EVEN_K:
            a = tl.load(a_block, eviction_policy="evict_first")
            b = tl.load(b_block, eviction_policy="evict_last")
        else:
            a = tl.load(
                a_block,
                boundary_check=(0, 1),
                padding_option="zero",
                eviction_policy="evict_first",
            )
            b = tl.load(
                b_block,
                boundary_check=(0, 1),
                padding_option="zero",
                eviction_policy="evict_last",
            )

        acc = tl.dot(a, b, acc, input_precision="tf32x3", out_dtype=tl.float32)

        a_block = tl.advance(a_block, (0, BLOCK_K))
        b_block = tl.advance(b_block, (BLOCK_K, 0))

    c_block = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    if EVEN_M and EVEN_N:
        tl.store(c_block, acc)
    else:
        tl.store(c_block, acc, boundary_check=(0, 1))


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8
    num_warps = 8
    num_stages = 3

    EVEN_M = (M % BLOCK_M) == 0
    EVEN_N = (N % BLOCK_N) == 0
    EVEN_K = (K % BLOCK_K) == 0

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _matmul_kernel[grid](
        a,
        b,
        output,
        M,
        N,
        K,
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
        NUM_STAGES=num_stages,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_K=EVEN_K,
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
            "num_warps": num_warps,
            "num_stages": num_stages,
            "input_precision": "tf32x3",
            "block_ptr": True,
            "warp_specialize": True,
            "b_eviction_policy": "evict_last",
            "EVEN_M": EVEN_M,
            "EVEN_N": EVEN_N,
            "EVEN_K": EVEN_K,
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
def _matmul_kernel(
    a,
    b,
    output,
    TILE_M: ConstInt,
    TILE_N: ConstInt,
    TILE_K: ConstInt,
    LOAD_LATENCY: ConstInt,
):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    num_k_tiles = ct.cdiv(a.shape[1], TILE_K)

    acc = ct.full((TILE_M, TILE_N), 0.0, dtype=np.float32)

    for k_tile in range(0, num_k_tiles):
        a_tile = ct.load(
            a,
            index=(bid_m, k_tile),
            shape=(TILE_M, TILE_K),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LATENCY,
        )
        b_tile = ct.load(
            b,
            index=(k_tile, bid_n),
            shape=(TILE_K, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=LOAD_LATENCY,
        )

        if a.dtype == ct.float32:
            a_hi = ct.astype(a_tile, ct.tfloat32)
            b_hi = ct.astype(b_tile, ct.tfloat32)

            a_res = ct.astype(
                ct.astype(a_tile, np.float32) - ct.astype(a_hi, np.float32),
                ct.tfloat32,
            )
            b_res = ct.astype(
                ct.astype(b_tile, np.float32) - ct.astype(b_hi, np.float32),
                ct.tfloat32,
            )

            acc = ct.mma(a_hi, b_hi, acc)
            acc = ct.mma(a_res, b_hi, acc)
            acc = ct.mma(a_hi, b_res, acc)
        else:
            acc = ct.mma(a_tile, b_tile, acc)

    ct.store(output, index=(bid_m, bid_n), tile=ct.astype(acc, a.dtype))


def run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]

    output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    stream = torch.cuda.current_stream()

    TILE_M = 128
    TILE_N = 256
    TILE_K = 64
    occupancy = 4
    load_latency = 10

    grid = (ct.cdiv(M, TILE_M), ct.cdiv(N, TILE_N), 1)
    ct.launch(
        stream,
        grid,
        _matmul_kernel,
        (a, b, output, TILE_M, TILE_N, TILE_K, load_latency),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE_M": TILE_M,
            "TILE_N": TILE_N,
            "TILE_K": TILE_K,
            "occupancy": occupancy,
            "load_latency": load_latency,
            "fp32_mma_input": "manual_tf32x3",
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
