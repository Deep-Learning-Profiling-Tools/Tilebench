```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv1d_kernel(input_ptr, kernel_ptr, output_ptr,
                   input_size, output_size, kernel_size,
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
        a = tl.load(input_ptr + a_idx, mask=a_mask, other=0.0)

        b_idx = k[:, None] - offs_n[None, :]
        b_mask = (b_idx >= 0) & (b_idx < kernel_size)
        b_safe = tl.where(b_mask, b_idx, 0)
        b = tl.load(kernel_ptr + b_safe, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, input_precision="tf32")

    out_idx = offs_m[:, None] * BLOCK_N + offs_n[None, :]
    out_mask = out_idx < output_size
    tl.store(output_ptr + out_idx, acc, mask=out_mask)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    K_TOTAL_PAD = triton.cdiv(kernel_size + BLOCK_N - 1, BLOCK_K) * BLOCK_K
    num_warps = 8
    num_stages = 2

    num_row_groups = triton.cdiv(output_size, BLOCK_N)
    grid = (triton.cdiv(num_row_groups, BLOCK_M),)

    _conv1d_kernel[grid](
        input, kernel, output,
        input_size, output_size, kernel_size,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        K_TOTAL_PAD=K_TOTAL_PAD,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
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


@ct.kernel
def _conv1d_kernel(input, kernel, output,
                   K_TOTAL_PAD: ConstInt,
                   BLOCK_M: ConstInt, BLOCK_N: ConstInt, BLOCK_K: ConstInt):
    bid = ct.bid(0)

    rows_1d = bid * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    cols_1d = ct.arange(BLOCK_N, dtype=np.int32)
    k_1d = ct.arange(BLOCK_K, dtype=np.int32)

    rows = rows_1d[:, None]
    cols = cols_1d[None, :]

    acc = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=np.float32)

    for k0 in range(0, K_TOTAL_PAD, BLOCK_K):
        kk_row = (k0 + k_1d)[None, :]
        a_idx = rows * BLOCK_N + kk_row
        a = ct.gather(input, a_idx, padding_value=0.0, check_bounds=True)

        kk_col = (k0 + k_1d)[:, None]
        b_idx = kk_col - cols
        b = ct.gather(kernel, b_idx, padding_value=0.0, check_bounds=True)

        acc = ct.mma(a, b, acc)

    out_idx = rows * BLOCK_N + cols
    ct.scatter(output, out_idx, ct.astype(acc, input.dtype), check_bounds=True)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int, **kwargs):
    output_size = input_size - kernel_size + 1
    output = torch.empty((output_size,), device=input.device, dtype=input.dtype)
    stream = torch.cuda.current_stream()

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64
    K_TOTAL_PAD = ((kernel_size + BLOCK_N - 1 + BLOCK_K - 1) // BLOCK_K) * BLOCK_K
    occupancy = 2

    num_row_groups = (output_size + BLOCK_N - 1) // BLOCK_N
    grid = ((num_row_groups + BLOCK_M - 1) // BLOCK_M, 1, 1)

    launch_kernel = _conv1d_kernel.with_hints(occupancy=occupancy)
    ct.launch(
        stream,
        grid,
        launch_kernel,
        (input, kernel, output, K_TOTAL_PAD, BLOCK_M, BLOCK_N, BLOCK_K),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "K_TOTAL_PAD": K_TOTAL_PAD,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
