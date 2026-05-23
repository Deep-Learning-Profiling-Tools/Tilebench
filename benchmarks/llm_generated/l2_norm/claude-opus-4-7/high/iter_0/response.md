Approach: Single-pass row-wise L2-norm. Treat the (batch, M, K) input as 2D (rows, K), launch one CTA per row, load the full row into a single tile (BLOCK_K = next pow2 of K, max K=10240 → 16384), compute sum-of-squares in fp32, then divide by max(‖x‖, eps) (matches `F.normalize`). num_warps scales with BLOCK_K.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _l2_norm_kernel(x_ptr, out_ptr, K, eps, stride_row,
                    BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    row_x = x_ptr + row * stride_row
    row_o = out_ptr + row * stride_row
    x = tl.load(row_x + offs, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(sum_sq)
    denom = tl.maximum(norm, eps)
    y = x / denom
    tl.store(row_o + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K < 512:
        BLOCK_K = 512
    if BLOCK_K <= 1024:
        num_warps = 4
    elif BLOCK_K <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    num_stages = 1

    grid = (M,)
    _l2_norm_kernel[grid](
        x_flat, out_flat, K, float(eps), x_flat.stride(0),
        BLOCK_K=BLOCK_K, num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_K": BLOCK_K,
        "num_warps": num_warps,
        "num_stages": num_stages,
    })
    return out_flat.reshape(shape)


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
def _l2_norm_kernel(x, out, eps: float, K: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    tile = ct.load(x, index=(row, 0), shape=(1, TILE),
                   padding_mode=ct.PaddingMode.ZERO)
    tf = ct.astype(tile, np.float32)
    sum_sq = ct.sum(tf * tf)
    norm = ct.sqrt(sum_sq)
    denom = ct.maximum(norm, eps)
    y = tf / denom
    ct.store(out, index=(row, 0), tile=ct.astype(y, x.dtype))


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1]).contiguous()
    M, K = x_flat.shape
    out_flat = torch.empty_like(x_flat)

    TILE = max(_next_pow2(K), 512)
    if TILE <= 1024:
        occupancy = 4
    elif TILE <= 4096:
        occupancy = 2
    else:
        occupancy = 1

    stream = torch.cuda.current_stream()
    grid = (M, 1, 1)
    kernel = _l2_norm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x_flat, out_flat, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return out_flat.reshape(shape)


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
