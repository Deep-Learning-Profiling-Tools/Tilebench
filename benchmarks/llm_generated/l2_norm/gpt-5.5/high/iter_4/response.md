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
    BLOCK_MAIN: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    row = tl.program_id(0)
    base = row * K

    offs0 = tl.arange(0, BLOCK_MAIN)
    mask0 = offs0 < K
    x0 = tl.load(
        x_ptr + base + offs0,
        mask=mask0,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    x0f = x0.to(tl.float32)
    sum_sq = tl.sum(x0f * x0f, axis=0)

    if K > BLOCK_MAIN:
        offs1 = tl.arange(0, BLOCK_TAIL)
        k1 = BLOCK_MAIN + offs1
        mask1 = k1 < K
        x1 = tl.load(
            x_ptr + base + k1,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
            eviction_policy="evict_first",
        )
        x1f = x1.to(tl.float32)
        sum_sq += tl.sum(x1f * x1f, axis=0)

    eps_f = sum_sq * 0.0 + eps
    norm = tl.sqrt_rn(sum_sq)
    denom = tl.maximum(norm, eps_f)
    rstd = tl.div_rn(1.0, denom)

    tl.store(
        out_ptr + base + offs0,
        x0 * rstd,
        mask=mask0,
        cache_modifier=".cs",
        eviction_policy="evict_first",
    )

    if K > BLOCK_MAIN:
        tl.store(
            out_ptr + base + k1,
            x1 * rstd,
            mask=mask1,
            cache_modifier=".cs",
            eviction_policy="evict_first",
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    rows = batch * M

    BLOCK_MAIN = 8192
    BLOCK_TAIL = 2048
    num_warps = 32
    num_stages = 3

    grid = (rows,)
    _l2_norm_kernel[grid](
        x,
        output,
        eps,
        K=K,
        BLOCK_MAIN=BLOCK_MAIN,
        BLOCK_TAIL=BLOCK_TAIL,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_MAIN": BLOCK_MAIN,
            "BLOCK_TAIL": BLOCK_TAIL,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOAD_CACHE_CG": 1,
            "STORE_CACHE_CS": 1,
            "STORE_INPUT_DTYPE": 1,
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


@ct.kernel(occupancy=1)
def _l2_norm_kernel(x, output, eps, TILE: ConstInt):
    row = ct.bid(0)
    m_size = x.shape[1]

    b = row // m_size
    m = row - b * m_size

    x_tile = ct.load(
        x,
        index=(b, m, 0),
        shape=(1, 1, TILE),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )
    x_f = ct.astype(x_tile, np.float32)

    sum_sq = ct.sum(x_f * x_f)
    eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
    norm = ct.sqrt(sum_sq)
    denom = ct.maximum(norm, eps_f)
    rstd = 1.0 / denom

    y = ct.astype(x_tile * rstd, x.dtype)
    ct.store(
        output,
        index=(b, m, 0),
        tile=y,
        latency=1,
        allow_tma=False,
    )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]

    TILE = 16384
    occupancy = 1

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    ct.launch(stream, grid, _l2_norm_kernel, (x, output, eps, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "TILE": TILE,
            "occupancy": occupancy,
            "STORE_INPUT_DTYPE": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
