```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _l2_norm_kernel(
    x_ptr,
    out_ptr,
    eps,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * K
    offs = tl.arange(0, BLOCK_K)

    o0 = offs
    m0 = o0 < K
    x0 = tl.load(x_ptr + base + o0, mask=m0, other=0.0, eviction_policy="evict_first")
    x0f = x0.to(tl.float32)
    sum_sq = tl.sum(x0f * x0f, axis=0)

    if K > BLOCK_K:
        o1 = BLOCK_K + offs
        m1 = o1 < K
        x1 = tl.load(x_ptr + base + o1, mask=m1, other=0.0, eviction_policy="evict_first")
        x1f = x1.to(tl.float32)
        sum_sq += tl.sum(x1f * x1f, axis=0)

    if K > 2 * BLOCK_K:
        o2 = 2 * BLOCK_K + offs
        m2 = o2 < K
        x2 = tl.load(x_ptr + base + o2, mask=m2, other=0.0, eviction_policy="evict_first")
        x2f = x2.to(tl.float32)
        sum_sq += tl.sum(x2f * x2f, axis=0)

    if K > 3 * BLOCK_K:
        o3 = 3 * BLOCK_K + offs
        m3 = o3 < K
        x3 = tl.load(x_ptr + base + o3, mask=m3, other=0.0, eviction_policy="evict_first")
        x3f = x3.to(tl.float32)
        sum_sq += tl.sum(x3f * x3f, axis=0)

    if K > 4 * BLOCK_K:
        o4 = 4 * BLOCK_K + offs
        m4 = o4 < K
        x4 = tl.load(x_ptr + base + o4, mask=m4, other=0.0, eviction_policy="evict_first")
        x4f = x4.to(tl.float32)
        sum_sq += tl.sum(x4f * x4f, axis=0)

    eps_f = sum_sq * 0.0 + eps
    norm = tl.sqrt_rn(sum_sq)
    denom = tl.maximum(norm, eps_f)
    rstd = tl.div_rn(1.0, denom)

    tl.store(
        out_ptr + base + o0,
        x0.to(tl.float32) * rstd,
        mask=m0,
        eviction_policy="evict_first",
    )

    if K > BLOCK_K:
        tl.store(
            out_ptr + base + o1,
            x1.to(tl.float32) * rstd,
            mask=m1,
            eviction_policy="evict_first",
        )

    if K > 2 * BLOCK_K:
        tl.store(
            out_ptr + base + o2,
            x2.to(tl.float32) * rstd,
            mask=m2,
            eviction_policy="evict_first",
        )

    if K > 3 * BLOCK_K:
        tl.store(
            out_ptr + base + o3,
            x3.to(tl.float32) * rstd,
            mask=m3,
            eviction_policy="evict_first",
        )

    if K > 4 * BLOCK_K:
        tl.store(
            out_ptr + base + o4,
            x4.to(tl.float32) * rstd,
            mask=m4,
            eviction_policy="evict_first",
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    rows = batch * M

    BLOCK_K = 2048
    MAX_BLOCKS = 5
    num_warps = 8
    num_stages = 3

    grid = (rows,)
    _l2_norm_kernel[grid](
        x,
        output,
        eps,
        K=K,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "MAX_BLOCKS": MAX_BLOCKS,
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
def _l2_norm_kernel(x, output, eps, TILE: ConstInt):
    row = ct.bid(0)
    m_size = x.shape[1]

    b = row // m_size
    m = row - b * m_size

    x0 = ct.load(
        x,
        index=(b, m, 0),
        shape=(1, 1, TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    x0f = ct.astype(x0, np.float32)
    sum_sq = ct.sum(x0f * x0f)

    if x.shape[2] > TILE:
        x1 = ct.load(
            x,
            index=(b, m, 1),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x1f = ct.astype(x1, np.float32)
        sum_sq = sum_sq + ct.sum(x1f * x1f)

    if x.shape[2] > 2 * TILE:
        x2 = ct.load(
            x,
            index=(b, m, 2),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x2f = ct.astype(x2, np.float32)
        sum_sq = sum_sq + ct.sum(x2f * x2f)

    if x.shape[2] > 3 * TILE:
        x3 = ct.load(
            x,
            index=(b, m, 3),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x3f = ct.astype(x3, np.float32)
        sum_sq = sum_sq + ct.sum(x3f * x3f)

    if x.shape[2] > 4 * TILE:
        x4 = ct.load(
            x,
            index=(b, m, 4),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x4f = ct.astype(x4, np.float32)
        sum_sq = sum_sq + ct.sum(x4f * x4f)

    eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
    norm = ct.sqrt(sum_sq)
    denom = ct.maximum(norm, eps_f)
    rstd = 1.0 / denom

    y0 = ct.astype(ct.astype(x0, np.float32) * rstd, x.dtype)
    ct.store(
        output,
        index=(b, m, 0),
        tile=y0,
        latency=1,
        allow_tma=False,
    )

    if x.shape[2] > TILE:
        y1 = ct.astype(ct.astype(x1, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 1),
            tile=y1,
            latency=1,
            allow_tma=False,
        )

    if x.shape[2] > 2 * TILE:
        y2 = ct.astype(ct.astype(x2, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 2),
            tile=y2,
            latency=1,
            allow_tma=False,
        )

    if x.shape[2] > 3 * TILE:
        y3 = ct.astype(ct.astype(x3, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 3),
            tile=y3,
            latency=1,
            allow_tma=False,
        )

    if x.shape[2] > 4 * TILE:
        y4 = ct.astype(ct.astype(x4, np.float32) * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 4),
            tile=y4,
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]

    TILE = 2048
    MAX_TILES = 5
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    kernel = _l2_norm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, eps, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "MAX_TILES": MAX_TILES,
            "occupancy": occupancy,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
