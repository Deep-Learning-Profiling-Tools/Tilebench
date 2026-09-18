Approach: tile-scaled elementwise multiply. Each block handles a 2D BLOCK_M×BLOCK_N tile; per element compute `(i//TILE_SIZE, j//TILE_SIZE)` to index into the scale matrix, then multiply in fp32 and cast back. Triton uses computed pointer loads; cuTile uses `ct.gather` to fetch per-element scales.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _wdq_kernel(X_ptr, S_ptr, Y_ptr,
                M, N, SM, SN,
                stride_xm, stride_xn,
                stride_sm, stride_sn,
                stride_ym, stride_yn,
                TILE_SIZE,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0)

    s_row = offs_m // TILE_SIZE
    s_col = offs_n // TILE_SIZE
    s_mask = (s_row[:, None] < SM) & (s_col[None, :] < SN)
    s_ptrs = S_ptr + s_row[:, None] * stride_sm + s_col[None, :] * stride_sn
    s = tl.load(s_ptrs, mask=s_mask, other=0)

    y = (x.to(tl.float32) * s.to(tl.float32)).to(x.dtype.element_ty)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    SM, SN = S.shape

    BLOCK_M = 128
    BLOCK_N = 128
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _wdq_kernel[grid](
        X, S, Y,
        M, N, SM, SN,
        X.stride(0), X.stride(1),
        S.stride(0), S.stride(1),
        Y.stride(0), Y.stride(1),
        TILE_SIZE,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "num_warps": num_warps, "num_stages": num_stages,
    })
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
def _wdq_kernel(X, S, Y, M, N, TILE_SIZE,
                BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x = ct.load(X, index=(bid_m, bid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO)

    offs_m = bid_m * BLOCK_M + ct.arange(BLOCK_M, dtype=np.int32)
    offs_n = bid_n * BLOCK_N + ct.arange(BLOCK_N, dtype=np.int32)
    s_row = (offs_m // TILE_SIZE)[:, None]
    s_col = (offs_n // TILE_SIZE)[None, :]

    s = ct.gather(S, (s_row, s_col), padding_value=0)

    y = ct.astype(ct.astype(x, np.float32) * ct.astype(s, np.float32), X.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=y)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    BLOCK_M = 128
    BLOCK_N = 128
    occupancy = 4

    grid = (ct.cdiv(M, BLOCK_M), ct.cdiv(N, BLOCK_N), 1)
    kernel = _wdq_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel, (X, S, Y, M, N, TILE_SIZE, BLOCK_M, BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "occupancy": occupancy,
    })
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
