Looking at iter 1:
- **Triton** failed to compile — likely the `x.dtype.element_ty` call (for a loaded tile, `x.dtype` is already the element dtype). Fix: just let `tl.store` auto-cast.
- **cuTile** got 92.5% on fp32 but only ~35% on fp16/bf16. The tile is too small for fp16/bf16 — half the bytes means half the work per CTA, so launch/grid overhead dominates. Solution: enlarge `BLOCK_N` to 256 and broadcast a `(1, 2)` scale tile up to `(128, 256)` via reshape+broadcast_to.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _wdq_kernel(X_ptr, S_ptr, Y_ptr,
                M, N,
                stride_xm, stride_xn,
                stride_sm, stride_sn,
                stride_ym, stride_yn,
                M_S, N_S,
                TILE_SIZE: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_x = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask_x, other=0)

    # Per-element scale-tile index (TILE_SIZE is constexpr; div is cheap & block-uniform)
    si = offs_m // TILE_SIZE
    sj = offs_n // TILE_SIZE
    mask_s = (si[:, None] < M_S) & (sj[None, :] < N_S)
    s_ptrs = S_ptr + si[:, None] * stride_sm + sj[None, :] * stride_sn
    s = tl.load(s_ptrs, mask=mask_s, other=0)

    y = x.to(tl.float32) * s.to(tl.float32)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_x)  # auto-casts to Y dtype


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)

    BLOCK_M = 128
    BLOCK_N = 256
    num_warps = 8
    num_stages = 2

    M_S = (M + TILE_SIZE - 1) // TILE_SIZE
    N_S = (N + TILE_SIZE - 1) // TILE_SIZE

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _wdq_kernel[grid](
        X, S, Y,
        M, N,
        X.stride(0), X.stride(1),
        S.stride(0), S.stride(1),
        Y.stride(0), Y.stride(1),
        M_S, N_S,
        TILE_SIZE=TILE_SIZE,
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

_OCCUPANCY = 2
_BLOCK_M = 128
_BLOCK_N = 256
_TILE = 128  # matches config.yaml TILE_SIZE


@ct.kernel(occupancy=_OCCUPANCY)
def _wdq_kernel(X, S, Y,
                BLOCK_M: ConstInt, BLOCK_N: ConstInt, TILE: ConstInt,
                SM: ConstInt, SN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x = ct.load(X, index=(bid_m, bid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO)

    # scale tile of shape (SM, SN) — one scalar per TILE x TILE sub-block
    s = ct.load(S, index=(bid_m, bid_n),
                shape=(SM, SN),
                padding_mode=ct.PaddingMode.ZERO)

    # broadcast (SM, SN) -> (BLOCK_M, BLOCK_N) via (SM, 1, SN, 1) -> (SM, TILE, SN, TILE)
    s4 = ct.reshape(s, (SM, 1, SN, 1))
    s4b = ct.broadcast_to(s4, (SM, TILE, SN, TILE))
    s_full = ct.reshape(s4b, (BLOCK_M, BLOCK_N))

    y = ct.astype(ct.astype(x, np.float32) * ct.astype(s_full, np.float32), X.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=y)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    assert TILE_SIZE == _TILE, f"This impl expects TILE_SIZE={_TILE}, got {TILE_SIZE}"
    SM = _BLOCK_M // _TILE  # 1
    SN = _BLOCK_N // _TILE  # 2

    grid = (ct.cdiv(M, _BLOCK_M), ct.cdiv(N, _BLOCK_N), 1)
    ct.launch(stream, grid, _wdq_kernel,
              (X, S, Y, _BLOCK_M, _BLOCK_N, _TILE, SM, SN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": _BLOCK_M, "BLOCK_N": _BLOCK_N,
        "occupancy": _OCCUPANCY,
    })
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Triton: fixed `x.dtype.element_ty` (invalid for loaded tiles) by relying on `tl.store` auto-cast; bumped tile to 128×256, num_warps=8 for better HBM saturation. cuTile: doubled `BLOCK_N` to 256 and broadcast a `(1,2)` scale tile to `(128,256)` so each CTA does 2× the work and the fp16/bf16 cases stop being launch-overhead-limited.
