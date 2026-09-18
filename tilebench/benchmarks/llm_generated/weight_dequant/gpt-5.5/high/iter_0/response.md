```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _weight_dequant_kernel(
    X,
    S,
    Y,
    M,
    N,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x_offsets = offs_m[:, None] * N + offs_n[None, :]

    x = tl.load(X + x_offsets, mask=mask, other=0.0).to(tl.float32)

    n_scale_cols = tl.cdiv(N, TILE_SIZE)

    if TILE_SIZE == BLOCK_N:
        scale_row = (pid_m * BLOCK_M) // TILE_SIZE
        scale_col = pid_n
        s = tl.load(S + scale_row * n_scale_cols + scale_col).to(tl.float32)
        y = x * s
    else:
        scale_rows = offs_m // TILE_SIZE
        scale_cols = offs_n // TILE_SIZE
        s_offsets = scale_rows[:, None] * n_scale_cols + scale_cols[None, :]
        s = tl.load(S + s_offsets, mask=mask, other=0.0).to(tl.float32)
        y = x * s

    tl.store(Y + x_offsets, y, mask=mask)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)

    BLOCK_M = 16
    BLOCK_N = 128
    num_warps = 8
    num_stages = 2

    grid = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    _weight_dequant_kernel[grid](
        X,
        S,
        Y,
        M,
        N,
        TILE_SIZE=TILE_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "TILE_SIZE": int(TILE_SIZE),
        }
    )
    return Y


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
def _weight_dequant_kernel(
    X,
    S,
    Y,
    TILE_SIZE: ConstInt,
    BLOCK_M: ConstInt,
    BLOCK_N: ConstInt,
):
    bid_n = ct.bid(0)
    bid_m = ct.bid(1)

    x_tile = ct.load(
        X,
        index=(bid_m, bid_n),
        shape=(BLOCK_M, BLOCK_N),
        padding_mode=ct.PaddingMode.ZERO,
        latency=1,
        allow_tma=False,
    )

    if TILE_SIZE == BLOCK_N:
        scale_row = (bid_m * BLOCK_M) // TILE_SIZE
        scale_col = bid_n
        s = ct.load(S, index=(scale_row, scale_col), shape=(), latency=1, allow_tma=False)
        y_tile = ct.astype(x_tile, np.float32) * ct.astype(s, np.float32)
    else:
        offs_m = bid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)[:, None]
        offs_n = bid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)[None, :]
        scale_rows = offs_m // TILE_SIZE
        scale_cols = offs_n // TILE_SIZE
        s_tile = ct.gather(S, (scale_rows, scale_cols), padding_value=0, latency=1)
        y_tile = ct.astype(x_tile, np.float32) * ct.astype(s_tile, np.float32)

    ct.store(
        Y,
        index=(bid_m, bid_n),
        tile=ct.astype(y_tile, X.dtype),
        latency=1,
        allow_tma=False,
    )


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    BLOCK_M = 16
    BLOCK_N = 128
    occupancy = 8

    grid = (ct.cdiv(N, BLOCK_N), ct.cdiv(M, BLOCK_M), 1)
    kernel = _weight_dequant_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (X, S, Y, TILE_SIZE, BLOCK_M, BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "occupancy": occupancy,
            "TILE_SIZE": int(TILE_SIZE),
        }
    )
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
