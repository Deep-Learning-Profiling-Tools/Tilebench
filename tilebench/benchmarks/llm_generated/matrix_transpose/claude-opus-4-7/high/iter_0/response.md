Approach: Both kernels use 2D `(BLOCK_M, BLOCK_N)` tiling. For Triton I `tl.load` an `[BM, BN]` block, then `tl.trans` and `tl.store` to the transposed location. For cuTile I `ct.load` a `[BM, BN]` tile with zero padding, `ct.transpose`, then `ct.store` to the swapped tile-space index — power-of-2 tile dims handle all shapes since n is always a multiple of 1024.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _transpose_kernel(x_ptr, out_ptr, M, N,
                      stride_xm, stride_xn,
                      stride_om, stride_on,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    in_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptrs, mask=in_mask)

    x_t = tl.trans(x)  # [BN, BM]

    o_ptrs = out_ptr + offs_n[:, None] * stride_om + offs_m[None, :] * stride_on
    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(o_ptrs, x_t, mask=out_mask)


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    M, N = x.shape
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)

    BLOCK_M = 64
    BLOCK_N = 64
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _transpose_kernel[grid](
        x, output, M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
                      "num_warps": num_warps, "num_stages": num_stages})
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
def _transpose_kernel(x, output, BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    tile = ct.load(x, index=(bid_m, bid_n), shape=(BM, BN),
                   padding_mode=ct.PaddingMode.ZERO)
    tile_t = ct.transpose(tile)
    ct.store(output, index=(bid_n, bid_m), tile=tile_t)


def run(x: torch.Tensor):
    if x.dim() != 2:
        raise ValueError("Input tensor for matrix_transpose must be 2D.")
    M, N = x.shape
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)
    stream = torch.cuda.current_stream()

    BM = 64
    BN = 64
    occupancy = 4

    grid = (ct.cdiv(M, BM), ct.cdiv(N, BN), 1)
    kernel = _transpose_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BM": BM, "BN": BN, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
