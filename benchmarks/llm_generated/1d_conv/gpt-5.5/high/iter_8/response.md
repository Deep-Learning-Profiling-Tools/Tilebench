```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _build_b_half_kernel(kernel_ptr, b_hi_ptr,
                         kernel_size,
                         total_elems,
                         BLOCK_N: tl.constexpr,
                         BUILD_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BUILD_BLOCK + tl.arange(0, BUILD_BLOCK)
    mask = offs < total_elems

    k = offs // BLOCK_N
    n = offs - k * BLOCK_N
    j = k - n

    valid = mask & (j >= 0) & (j < kernel_size)
    j_safe = tl.where(valid, j, 0)

    vals = tl.load(kernel_ptr + j_safe, mask=valid, other=0.0)
    tl.store(b_hi_ptr + offs, vals.to(tl.float16), mask=mask)


@triton.jit
def _build_b_split_kernel(kernel_ptr, b_hi_ptr, b_lo_ptr,
                          kernel_size,
                          total_elems,
                          BLOCK_N: tl.constexpr,
                          BUILD_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BUILD_BLOCK + tl.arange(0, BUILD_BLOCK)
    mask = offs < total_elems

    k = offs // BLOCK_N
    n = offs - k * BLOCK_N
    j = k - n

    valid = mask & (j >= 0) & (j < kernel_size)
    j_safe = tl.where(valid, j, 0)

    vals_f = tl.load(kernel_ptr + j_safe, mask=valid, other=0.0).to(tl.float32)
    vals_hi = vals_f.to(tl.float16)
    vals_lo = (vals_f - vals_hi.to(tl.float32)).to(tl.float16)

    tl.store(b_hi_ptr + offs, vals_hi, mask=mask)
    tl.store(b_lo_ptr + offs, vals_lo, mask=mask)


@triton.jit
def _conv1d_mma_dense_half_kernel(input_ptr, b_hi_ptr, output_ptr,
                                  input_size, output_size,
                                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                  BLOCK_K: tl.constexpr, K_TOTAL_PAD: tl.constexpr):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_row_groups = tl.cdiv(output_size, BLOCK_N)
    row_valid = offs_m < num_row_groups

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_TOTAL_PAD, BLOCK_K):
        k = k0 + offs_k

        a_idx = offs_m[:, None] * BLOCK_N + k[None, :]
        a_mask = row_valid[:, None] & (a_idx < input_size)
        a = tl.load(
            input_ptr + a_idx,
            mask=a_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float16)

        b_idx = k[:, None] * BLOCK_N + offs_n[None, :]
        b = tl.load(
            b_hi_ptr + b_idx,
            eviction_policy="evict_last",
        )

        acc = tl.dot(a, b, acc)

    out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_idx < output_size
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def _conv1d_mma_dense_split4_kernel(input_ptr, b_hi_ptr, b_lo_ptr, output_ptr,
                                    input_size, output_size,
                                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                    BLOCK_K: tl.constexpr, K_TOTAL_PAD: tl.constexpr):
    pid = tl.program_id(0)

    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    num_row_groups = tl.cdiv(output_size, BLOCK_N)
    row_valid = offs_m < num_row_groups

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_TOTAL_PAD, BLOCK_K):
        k = k0 + offs_k

        a_idx = offs_m[:, None] * BLOCK_N + k[None, :]
        a_mask = row_valid[:, None] & (a_idx < input_size)
        a_f = tl.load(
            input_ptr + a_idx,
            mask=a_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)

        a_hi = a_f.to(tl.float16)
        a_lo = (a_f - a_hi.to(tl.float32)).to(tl.float16)

        b_idx = k[:, None] * BLOCK_N + offs_n[None, :]

        b_hi = tl.load(
            b_hi_ptr + b_idx,
            eviction_policy="evict_last",
        )
        acc = tl.dot(a_hi, b_hi, acc)
        acc = tl.dot(a_lo, b_hi, acc)

        b_lo = tl.load(
            b_lo_ptr + b_idx,
            eviction_policy="evict_last",
        )
        acc = tl.dot(a_hi, b_lo, acc)
        acc = tl.dot(a_lo, b_lo, acc)

    out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_idx < output_size
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64
    K_TOTAL_PAD = triton.cdiv(kernel_size + BLOCK_N - 1, BLOCK_K) * BLOCK_K
    MMA_BLOCK_OUT = BLOCK_M * BLOCK_N

    BUILD_BLOCK = 1024
    build_num_warps = 4
    build_num_stages = 2

    mma_num_warps = 4
    mma_num_stages = 3

    b_hi = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=torch.float16)
    total_b = K_TOTAL_PAD * BLOCK_N
    build_grid = (triton.cdiv(total_b, BUILD_BLOCK),)

    grid = (triton.cdiv(output_size, MMA_BLOCK_OUT),)

    if input.dtype == torch.float32:
        b_lo = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=torch.float16)
        _build_b_split_kernel[build_grid](
            kernel, b_hi, b_lo,
            kernel_size,
            total_b,
            BLOCK_N=BLOCK_N,
            BUILD_BLOCK=BUILD_BLOCK,
            num_warps=build_num_warps,
            num_stages=build_num_stages,
        )
        _conv1d_mma_dense_split4_kernel[grid](
            input, b_hi, b_lo, output,
            input_size, output_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            K_TOTAL_PAD=K_TOTAL_PAD,
            num_warps=mma_num_warps,
            num_stages=mma_num_stages,
        )
    else:
        _build_b_half_kernel[build_grid](
            kernel, b_hi,
            kernel_size,
            total_b,
            BLOCK_N=BLOCK_N,
            BUILD_BLOCK=BUILD_BLOCK,
            num_warps=build_num_warps,
            num_stages=build_num_stages,
        )
        _conv1d_mma_dense_half_kernel[grid](
            input, b_hi, output,
            input_size, output_size,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            K_TOTAL_PAD=K_TOTAL_PAD,
            num_warps=mma_num_warps,
            num_stages=mma_num_stages,
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "MMA_BLOCK_OUT": MMA_BLOCK_OUT,
        "KERNEL_SIZE": kernel_size,
        "BUILD_BLOCK": BUILD_BLOCK,
        "build_num_warps": build_num_warps,
        "build_num_stages": build_num_stages,
        "mma_num_warps": mma_num_warps,
        "mma_num_stages": mma_num_stages,
        "BMAT_DTYPE_FP16": True,
        "FP32_FP16_SPLIT4_TC": True,
        "DENSE_B_ALL": True,
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


@ct.kernel
def _build_b_half_kernel(kernel, b_hi,
                         KERNEL_SIZE: ConstInt,
                         BLOCK_N: ConstInt,
                         BUILD_M: ConstInt,
                         BUILD_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    rows = bid_m * BUILD_M + ct.arange(BUILD_M, dtype=np.int32)[:, None]
    cols = bid_n * BUILD_N + ct.arange(BUILD_N, dtype=np.int32)[None, :]

    j = rows - cols
    vals = ct.gather(kernel, j, padding_value=0.0, check_bounds=True, latency=1)
    vals_hi = ct.astype(ct.astype(vals, np.float32), np.float16)

    ct.store(
        b_hi,
        index=(bid_m, bid_n),
        tile=vals_hi,
        latency=1,
        allow_tma=False,
    )


@ct.kernel
def _build_b_split_kernel(kernel, b_hi, b_lo,
                          KERNEL_SIZE: ConstInt,
                          BLOCK_N: ConstInt,
                          BUILD_M: ConstInt,
                          BUILD_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    rows = bid_m * BUILD_M + ct.arange(BUILD_M, dtype=np.int32)[:, None]
    cols = bid_n * BUILD_N + ct.arange(BUILD_N, dtype=np.int32)[None, :]

    j = rows - cols
    vals_f = ct.astype(
        ct.gather(kernel, j, padding_value=0.0, check_bounds=True, latency=1),
        np.float32,
    )
    vals_hi = ct.astype(vals_f, np.float16)
    vals_lo = ct.astype(vals_f - ct.astype(vals_hi, np.float32), np.float16)

    ct.store(
        b_hi,
        index=(bid_m, bid_n),
        tile=vals_hi,
        latency=1,
        allow_tma=False,
    )
    ct.store(
        b_lo,
        index=(bid_m, bid_n),
        tile=vals_lo,
        latency=1,
        allow_tma=False,
    )


@ct.kernel
def _conv1d_mma_dense_half_kernel(input, b_hi, output,
                                  K_TOTAL_PAD: ConstInt,
                                  BLOCK_M: ConstInt,
                                  BLOCK_N: ConstInt,
                                  BLOCK_K: ConstInt):
    bid = ct.bid(0)

    rows_1d = bid * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    cols_1d = ct.arange(BLOCK_N, dtype=np.int32)
    k_1d = ct.arange(BLOCK_K, dtype=np.int32)

    rows = rows_1d[:, None]
    cols = cols_1d[None, :]

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)

    for k0 in range(0, K_TOTAL_PAD, BLOCK_K):
        kk = (k0 + k_1d)[None, :]
        a_idx = rows * BLOCK_N + kk
        a = ct.astype(
            ct.gather(input, a_idx, padding_value=0.0, check_bounds=True, latency=1),
            np.float16,
        )

        b = ct.load(
            b_hi,
            index=(k0 // BLOCK_K, 0),
            shape=(BLOCK_K, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )

        acc = ct.mma(a, b, acc)

    out_idx = rows * BLOCK_N + cols
    ct.scatter(output, out_idx, ct.astype(acc, input.dtype), check_bounds=True, latency=1)


@ct.kernel
def _conv1d_mma_dense_split4_kernel(input, b_hi, b_lo, output,
                                    K_TOTAL_PAD: ConstInt,
                                    BLOCK_M: ConstInt,
                                    BLOCK_N: ConstInt,
                                    BLOCK_K: ConstInt):
    bid = ct.bid(0)

    rows_1d = bid * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    cols_1d = ct.arange(BLOCK_N, dtype=np.int32)
    k_1d = ct.arange(BLOCK_K, dtype=np.int32)

    rows = rows_1d[:, None]
    cols = cols_1d[None, :]

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)

    for k0 in range(0, K_TOTAL_PAD, BLOCK_K):
        kk = (k0 + k_1d)[None, :]
        a_idx = rows * BLOCK_N + kk

        a_f = ct.astype(
            ct.gather(input, a_idx, padding_value=0.0, check_bounds=True, latency=1),
            np.float32,
        )
        a_hi = ct.astype(a_f, np.float16)
        a_lo = ct.astype(a_f - ct.astype(a_hi, np.float32), np.float16)

        bhi = ct.load(
            b_hi,
            index=(k0 // BLOCK_K, 0),
            shape=(BLOCK_K, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        acc = ct.mma(a_hi, bhi, acc)
        acc = ct.mma(a_lo, bhi, acc)

        blo = ct.load(
            b_lo,
            index=(k0 // BLOCK_K, 0),
            shape=(BLOCK_K, BLOCK_N),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        acc = ct.mma(a_hi, blo, acc)
        acc = ct.mma(a_lo, blo, acc)

    out_idx = rows * BLOCK_N + cols
    ct.scatter(output, out_idx, ct.astype(acc, input.dtype), check_bounds=True, latency=1)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 64
    K_TOTAL_PAD = ((kernel_size + BLOCK_N - 1 + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    MMA_BLOCK_OUT = BLOCK_M * BLOCK_N

    BUILD_M = BLOCK_K
    BUILD_N = BLOCK_N

    occupancy_build = 4
    occupancy_mma = 1

    b_hi = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=torch.float16)

    build_grid = (
        (K_TOTAL_PAD + BUILD_M - 1) // BUILD_M,
        (BLOCK_N + BUILD_N - 1) // BUILD_N,
        1,
    )
    grid = ((output_size + MMA_BLOCK_OUT - 1) // MMA_BLOCK_OUT, 1, 1)

    if input.dtype == torch.float32:
        b_lo = torch.empty((K_TOTAL_PAD, BLOCK_N), device=input.device, dtype=torch.float16)

        ct.launch(
            stream,
            build_grid,
            _build_b_split_kernel.with_hints(occupancy=occupancy_build),
            (kernel, b_hi, b_lo, kernel_size, BLOCK_N, BUILD_M, BUILD_N),
        )
        ct.launch(
            stream,
            grid,
            _conv1d_mma_dense_split4_kernel.with_hints(occupancy=occupancy_mma),
            (input, b_hi, b_lo, output, K_TOTAL_PAD, BLOCK_M, BLOCK_N, BLOCK_K),
        )
    else:
        ct.launch(
            stream,
            build_grid,
            _build_b_half_kernel.with_hints(occupancy=occupancy_build),
            (kernel, b_hi, kernel_size, BLOCK_N, BUILD_M, BUILD_N),
        )
        ct.launch(
            stream,
            grid,
            _conv1d_mma_dense_half_kernel.with_hints(occupancy=occupancy_mma),
            (input, b_hi, output, K_TOTAL_PAD, BLOCK_M, BLOCK_N, BLOCK_K),
        )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "MMA_BLOCK_OUT": MMA_BLOCK_OUT,
        "KERNEL_SIZE": kernel_size,
        "BUILD_M": BUILD_M,
        "BUILD_N": BUILD_N,
        "occupancy_build": occupancy_build,
        "occupancy_mma": occupancy_mma,
        "BMAT_DTYPE_FP16": True,
        "FP32_FP16_SPLIT4_TC": True,
        "DENSE_B_ALL": True,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
