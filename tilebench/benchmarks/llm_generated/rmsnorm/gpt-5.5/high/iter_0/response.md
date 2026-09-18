```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    K,
    eps,
    x_stride_b,
    x_stride_m,
    x_stride_k,
    w_stride_k,
    out_stride_b,
    out_stride_m,
    out_stride_k,
    M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // M
    m = row - b * M

    x_row = x_ptr + b * x_stride_b + m * x_stride_m
    out_row = out_ptr + b * out_stride_b + m * out_stride_m

    offs = tl.arange(0, BLOCK_N)
    partial = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for start in tl.range(0, K, BLOCK_N, num_stages=LOOP_STAGES):
        k = start + offs
        mask = k < K
        x = tl.load(
            x_row + k * x_stride_k,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        partial += x * x

    sum_sq = tl.sum(partial, axis=0)
    rstd = tl.rsqrt(sum_sq / K + eps)

    for start in tl.range(0, K, BLOCK_N, num_stages=LOOP_STAGES):
        k = start + offs
        mask = k < K
        x = tl.load(
            x_row + k * x_stride_k,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        w = tl.load(
            w_ptr + k * w_stride_k,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        y = x * rstd * w
        tl.store(out_row + k * out_stride_k, y, mask=mask)


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    n_rows = batch * M

    BLOCK_N = 1024
    num_warps = 4
    num_stages = 4

    grid = (n_rows,)
    _rmsnorm_kernel[grid](
        x,
        rms_w,
        output,
        K,
        eps,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rms_w.stride(0),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        M=M,
        BLOCK_N=BLOCK_N,
        LOOP_STAGES=num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
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


@ct.kernel
def _rmsnorm_kernel(x, rms_w, output, eps, K: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    M = x.shape[1]

    b = row // M
    m = row - b * M
    num_tiles = ct.cdiv(K, TILE)

    acc = ct.full((1, 1, TILE), 0.0, dtype=np.float32)

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(b, m, j),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj = ct.astype(xj, np.float32)
        acc = acc + xj * xj

    sum_sq = ct.sum(acc, axis=2, keepdims=True)
    rstd = ct.rsqrt(sum_sq / K + eps)

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(b, m, j),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj = ct.astype(xj, np.float32)

        wj = ct.load(
            rms_w,
            index=(j,),
            shape=(TILE,),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        wj = ct.astype(wj, np.float32)
        wj = ct.reshape(wj, (1, 1, TILE))

        y = xj * rstd * wj
        ct.store(
            output,
            index=(b, m, j),
            tile=ct.astype(y, x.dtype),
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    n_rows = batch * M

    TILE = 1024
    occupancy = 8

    stream = torch.cuda.current_stream()
    grid = (n_rows, 1, 1)
    kernel = _rmsnorm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, rms_w, output, eps, K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
