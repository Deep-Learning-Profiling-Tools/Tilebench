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
    M,
    K,
    sx_b,
    sx_m,
    sx_k,
    so_b,
    so_m,
    so_k,
    BLOCK_K: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    b = row // M
    m = row - b * M

    x_base = b * sx_b + m * sx_m
    o_base = b * so_b + m * so_m

    offs = tl.arange(0, BLOCK_K)

    sum_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K, num_stages=LOOP_STAGES):
        k = k0 + offs
        mask = k < K
        x = tl.load(x_ptr + x_base + k * sx_k, mask=mask, other=0.0).to(tl.float32)
        sum_vec += x * x

    sum_sq = tl.sum(sum_vec, axis=0)
    eps_f = sum_sq * 0.0 + eps
    norm = tl.sqrt_rn(sum_sq)
    denom = tl.maximum(norm, eps_f)
    rstd = tl.div_rn(1.0, denom)

    for k0 in tl.range(0, K, BLOCK_K, num_stages=LOOP_STAGES):
        k = k0 + offs
        mask = k < K
        x = tl.load(x_ptr + x_base + k * sx_k, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd
        tl.store(out_ptr + o_base + k * so_k, y, mask=mask)


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]
    rows = batch * M

    BLOCK_K = 1024
    num_warps = 4
    num_stages = 3
    LOOP_STAGES = 3

    grid = (rows,)
    _l2_norm_kernel[grid](
        x,
        output,
        eps,
        M,
        K,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_K=BLOCK_K,
        LOOP_STAGES=LOOP_STAGES,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_K": BLOCK_K,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "LOOP_STAGES": LOOP_STAGES,
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
def _l2_norm_kernel(x, output, eps, K: ConstInt, TILE: ConstInt):
    row = ct.bid(0)
    m_size = x.shape[1]

    b = row // m_size
    m = row - b * m_size

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
        xj_f = ct.astype(xj, np.float32)
        acc = acc + xj_f * xj_f

    sum_sq = ct.sum(acc)
    eps_f = sum_sq * 0.0 + eps
    norm = ct.sqrt(sum_sq)
    denom = ct.maximum(norm, eps_f)
    rstd = 1.0 / denom

    for j in range(0, num_tiles):
        xj = ct.load(
            x,
            index=(b, m, j),
            shape=(1, 1, TILE),
            padding_mode=ct.PaddingMode.ZERO,
            latency=1,
            allow_tma=False,
        )
        xj_f = ct.astype(xj, np.float32)
        y = ct.astype(xj_f * rstd, x.dtype)
        ct.store(
            output,
            index=(b, m, j),
            tile=y,
            latency=1,
            allow_tma=False,
        )


def run(x: torch.Tensor, eps: float = 1e-6, **kwargs) -> torch.Tensor:
    output = torch.empty_like(x)

    batch = x.shape[0]
    M = x.shape[1]
    K = x.shape[2]

    TILE = 1024
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (batch * M, 1, 1)
    kernel = _l2_norm_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (x, output, eps, K, TILE))

    _LAST_CFG.clear()
    _LAST_CFG.update({"TILE": TILE, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
