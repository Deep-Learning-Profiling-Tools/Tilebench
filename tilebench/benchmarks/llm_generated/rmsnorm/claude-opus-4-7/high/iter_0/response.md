```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr,
                    stride_row,
                    N, eps,
                    BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_row
    o_row = out_ptr + row * stride_row
    cols = tl.arange(0, BLOCK_N)

    # Pass 1: sum of squares
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + cols
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    rstd = tl.math.rsqrt(tl.sum(acc, axis=0) / N + eps)

    # Pass 2: normalize and scale
    for n0 in range(0, N, BLOCK_N):
        offs = n0 + cols
        mask = offs < N
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(o_row + offs, y, mask=mask)


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)
    batch, M, K = x.shape
    n_rows = batch * M

    x2 = x.reshape(n_rows, K)
    o2 = output.reshape(n_rows, K)
    stride_row = x2.stride(0)

    BLOCK_N = 2048 if K >= 2048 else triton.next_power_of_2(K)
    if BLOCK_N < 256:
        BLOCK_N = 256
    num_warps = 8
    num_stages = 2

    grid = (n_rows,)
    _rmsnorm_kernel[grid](
        x2, rms_w, o2,
        stride_row,
        K, eps,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({"BLOCK_N": BLOCK_N,
                      "num_warps": num_warps,
                      "num_stages": num_stages})
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
def _rmsnorm_kernel(x, w, out, eps: float, N: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    num_tiles = ct.cdiv(N, TILE)

    # Pass 1: sum of squares in fp32
    acc = ct.full((1, TILE), 0.0, dtype=np.float32)
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        acc = acc + xj * xj

    mean_sq = ct.sum(acc, axis=1, keepdims=True) / N
    rstd = ct.rsqrt(mean_sq + eps)

    # Pass 2: normalize and scale
    for j in range(0, num_tiles):
        xj = ct.astype(
            ct.load(x, index=(row, j), shape=(1, TILE),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        wj = ct.astype(
            ct.load(w, index=(j,), shape=(TILE,),
                    padding_mode=ct.PaddingMode.ZERO),
            np.float32,
        )
        yj = xj * rstd * wj[None, :]
        ct.store(out, index=(row, j), tile=ct.astype(yj, x.dtype))


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)
    batch, M, K = x.shape
    n_rows = batch * M

    x2 = x.reshape(n_rows, K)
    o2 = output.reshape(n_rows, K)

    # Pick TILE: power of 2, at least 256
    TILE = 1
    while TILE < K and TILE < 2048:
        TILE *= 2
    if TILE < 256:
        TILE = 256
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (n_rows, 1, 1)
    kernel = _rmsnorm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x2, rms_w, o2, float(eps), K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Two-pass RMSNorm: one CTA per row, tile the K dimension with `BLOCK_N=2048` (Triton) / `TILE=2048` (cuTile), accumulate sum-of-squares in fp32, then re-stream x and rms_w to write the normalized output.
