```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _process_one_row(
    x_ptr,
    out_ptr,
    eps,
    row,
    K: tl.constexpr,
    BLOCK_MAIN: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
):
    base = row * K
    offs0 = tl.arange(0, BLOCK_MAIN)

    if K == BLOCK_MAIN + BLOCK_TAIL:
        x0 = tl.load(x_ptr + base + offs0, cache_modifier=".cg")
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        offs1 = tl.arange(0, BLOCK_TAIL)
        k1 = BLOCK_MAIN + offs1
        x1 = tl.load(x_ptr + base + k1, cache_modifier=".cg")
        x1_f = x1.to(tl.float32)
        sum_sq += tl.sum(x1_f * x1_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(out_ptr + base + offs0, x0 * rstd, cache_modifier=".cs")
        tl.store(out_ptr + base + k1, x1 * rstd, cache_modifier=".cs")

    elif K > BLOCK_MAIN:
        x0 = tl.load(x_ptr + base + offs0, cache_modifier=".cg")
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        offs1 = tl.arange(0, BLOCK_TAIL)
        k1 = BLOCK_MAIN + offs1
        mask1 = k1 < K
        x1 = tl.load(
            x_ptr + base + k1,
            mask=mask1,
            other=0.0,
            cache_modifier=".cg",
        )
        x1_f = x1.to(tl.float32)
        sum_sq += tl.sum(x1_f * x1_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(out_ptr + base + offs0, x0 * rstd, cache_modifier=".cs")
        tl.store(
            out_ptr + base + k1,
            x1 * rstd,
            mask=mask1,
            cache_modifier=".cs",
        )

    elif K == BLOCK_MAIN:
        x0 = tl.load(x_ptr + base + offs0, cache_modifier=".cg")
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(out_ptr + base + offs0, x0 * rstd, cache_modifier=".cs")

    else:
        mask0 = offs0 < K
        x0 = tl.load(
            x_ptr + base + offs0,
            mask=mask0,
            other=0.0,
            cache_modifier=".cg",
        )
        x0_f = x0.to(tl.float32)
        sum_sq = tl.sum(x0_f * x0_f, axis=0)

        eps_f = sum_sq * 0.0 + eps
        rstd = tl.rsqrt(tl.maximum(sum_sq, eps_f * eps_f))

        tl.store(
            out_ptr + base + offs0,
            x0 * rstd,
            mask=mask0,
            cache_modifier=".cs",
        )


@triton.jit
def _l2_norm_kernel(
    x_ptr,
    out_ptr,
    eps,
    K: tl.constexpr,
    BLOCK_MAIN: tl.constexpr,
    BLOCK_TAIL: tl.constexpr,
    GRID_ROWS: tl.constexpr,
    DO_SECOND_ROW: tl.constexpr,
):
    pid = tl.program_id(0)

    _process_one_row(
        x_ptr,
        out_ptr,
        eps,
        pid,
        K,
        BLOCK_MAIN,
        BLOCK_TAIL,
    )

    if DO_SECOND_ROW:
        _process_one_row(
            x_ptr,
            out_ptr,
            eps,
            pid + GRID_ROWS,
            K,
            BLOCK_MAIN,
            BLOCK_TAIL,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    rows = batch * M

    BLOCK_MAIN = 8192
    BLOCK_TAIL = 2048
    num_warps = 16
    num_stages = 3

    if rows % 2 == 0:
        ROWS_PER_CTA = 2
        grid_rows = rows // 2
        DO_SECOND_ROW = True
    else:
        ROWS_PER_CTA = 1
        grid_rows = rows
        DO_SECOND_ROW = False

    grid = (grid_rows,)
    _l2_norm_kernel[grid](
        x,
        output,
        eps,
        K=K,
        BLOCK_MAIN=BLOCK_MAIN,
        BLOCK_TAIL=BLOCK_TAIL,
        GRID_ROWS=grid_rows,
        DO_SECOND_ROW=DO_SECOND_ROW,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_MAIN": BLOCK_MAIN,
            "BLOCK_TAIL": BLOCK_TAIL,
            "ROWS_PER_CTA": ROWS_PER_CTA,
            "GRID_ROWS": grid_rows,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOAD_CACHE_CG": 1,
            "STORE_CACHE_CS": 1,
            "KEEP_INPUT_DTYPE": 1,
            "MATH_RSQRT": 1,
            "FULL_K_FASTPATH": 1,
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
def _l2_norm_kernel(
    x,
    output,
    eps,
    BLOCK_MAIN: ConstInt,
    BLOCK_TAIL: ConstInt,
    TAIL_INDEX: ConstInt,
):
    row = ct.bid(0)
    m_size = x.shape[1]
    k_size = x.shape[2]

    b = row // m_size
    m = row - b * m_size

    if k_size == BLOCK_MAIN + BLOCK_TAIL:
        x0 = ct.load(
            x,
            index=(b, m, 0),
            shape=(1, 1, BLOCK_MAIN),
            latency=1,
            allow_tma=False,
        )
        x0_f = ct.astype(x0, np.float32)
        sum_sq = ct.sum(x0_f * x0_f)

        x1 = ct.load(
            x,
            index=(b, m, TAIL_INDEX),
            shape=(1, 1, BLOCK_TAIL),
            latency=1,
            allow_tma=False,
        )
        x1_f = ct.astype(x1, np.float32)
        sum_sq = sum_sq + ct.sum(x1_f * x1_f)

        eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
        rstd = ct.rsqrt(ct.maximum(sum_sq, eps_f * eps_f))

        y0 = ct.astype(x0_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 0),
            tile=y0,
            latency=1,
            allow_tma=False,
        )

        y1 = ct.astype(x1_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, TAIL_INDEX),
            tile=y1,
            latency=1,
            allow_tma=False,
        )

    elif k_size > BLOCK_MAIN:
        x0 = ct.load(
            x,
            index=(b, m, 0),
            shape=(1, 1, BLOCK_MAIN),
            latency=1,
            allow_tma=False,
        )
        x0_f = ct.astype(x0, np.float32)
        sum_sq = ct.sum(x0_f * x0_f)

        x1 = ct.load(
            x,
            index=(b, m, TAIL_INDEX),
            shape=(1, 1, BLOCK_TAIL),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x1_f = ct.astype(x1, np.float32)
        sum_sq = sum_sq + ct.sum(x1_f * x1_f)

        eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
        rstd = ct.rsqrt(ct.maximum(sum_sq, eps_f * eps_f))

        y0 = ct.astype(x0_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 0),
            tile=y0,
            latency=1,
            allow_tma=False,
        )

        y1 = ct.astype(x1_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, TAIL_INDEX),
            tile=y1,
            latency=1,
            allow_tma=False,
        )

    elif k_size == BLOCK_MAIN:
        x0 = ct.load(
            x,
            index=(b, m, 0),
            shape=(1, 1, BLOCK_MAIN),
            latency=1,
            allow_tma=False,
        )
        x0_f = ct.astype(x0, np.float32)
        sum_sq = ct.sum(x0_f * x0_f)

        eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
        rstd = ct.rsqrt(ct.maximum(sum_sq, eps_f * eps_f))

        y0 = ct.astype(x0_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 0),
            tile=y0,
            latency=1,
            allow_tma=False,
        )

    else:
        x0 = ct.load(
            x,
            index=(b, m, 0),
            shape=(1, 1, BLOCK_MAIN),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        x0_f = ct.astype(x0, np.float32)
        sum_sq = ct.sum(x0_f * x0_f)

        eps_f = ct.astype(sum_sq * 0.0 + eps, np.float32)
        rstd = ct.rsqrt(ct.maximum(sum_sq, eps_f * eps_f))

        y0 = ct.astype(x0_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, 0),
            tile=y0,
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]

    BLOCK_MAIN = 8192
    BLOCK_TAIL = 2048
    TAIL_INDEX = 4
    occupancy = 1

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    ct.launch(
        stream,
        grid,
        _l2_norm_kernel,
        (x, output, eps, BLOCK_MAIN, BLOCK_TAIL, TAIL_INDEX),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_MAIN": BLOCK_MAIN,
            "BLOCK_TAIL": BLOCK_TAIL,
            "TAIL_INDEX": TAIL_INDEX,
            "occupancy": occupancy,
            "SPLIT_TILE": 1,
            "MATH_RSQRT": 1,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
