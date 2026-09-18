Looking at the feedback:
- Triton iter 2 (67.9%) — fp32 at 55.7% drags down the avg; tile may be too large for fp32 register pressure. Try smaller tile + deeper pipelining.
- Cutile iter 2 regressed (54.6% → 9.5%) — the broadcast-reshape path was much slower than the simple 128×128 = one-scalar-load approach. Revert to iter 1 design and increase occupancy.

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

    si = offs_m // TILE_SIZE
    sj = offs_n // TILE_SIZE
    mask_s = (si[:, None] < M_S) & (sj[None, :] < N_S)
    s_ptrs = S_ptr + si[:, None] * stride_sm + sj[None, :] * stride_sn
    s = tl.load(s_ptrs, mask=mask_s, other=0)

    y = x.to(tl.float32) * s.to(tl.float32)

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, y, mask=mask_x)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)

    # Smaller per-program tile area than iter 2 (was 128×256) to reduce
    # register pressure for fp32 (the worst-performing dtype). Same total
    # work but more in-flight programs to hide HBM latency.
    BLOCK_M = 64
    BLOCK_N = 256
    num_warps = 8
    num_stages = 4

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

# Reverting to iter-1 design (one block == one scale tile, single scalar load
# of S) which gave 54.6%. Bump occupancy from 2 -> 4 to better hide HBM
# latency for this purely bandwidth-bound kernel.
_OCCUPANCY = 4
_BLOCK_M = 128
_BLOCK_N = 128


@ct.kernel(occupancy=_OCCUPANCY)
def _wdq_kernel(X, S, Y, BLOCK_M: ConstInt, BLOCK_N: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x = ct.load(X, index=(bid_m, bid_n),
                shape=(BLOCK_M, BLOCK_N),
                padding_mode=ct.PaddingMode.ZERO)

    # BLOCK_M == BLOCK_N == TILE_SIZE: one scalar per block.
    s = ct.load(S, index=(bid_m, bid_n), shape=())

    y = ct.astype(ct.astype(x, np.float32) * ct.astype(s, np.float32), X.dtype)
    ct.store(Y, index=(bid_m, bid_n), tile=y)


def run(X: torch.Tensor, S: torch.Tensor, M: int, N: int, TILE_SIZE: int, **kwargs):
    Y = torch.empty_like(X)
    stream = torch.cuda.current_stream()

    assert TILE_SIZE == _BLOCK_M, f"This impl expects TILE_SIZE={_BLOCK_M}, got {TILE_SIZE}"

    grid = (ct.cdiv(M, _BLOCK_M), ct.cdiv(N, _BLOCK_N), 1)
    ct.launch(stream, grid, _wdq_kernel, (X, S, Y, _BLOCK_M, _BLOCK_N))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": _BLOCK_M, "BLOCK_N": _BLOCK_N,
        "occupancy": _OCCUPANCY,
    })
    return Y


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: Triton — shrink tile to 64×256 with `num_stages=4` to ease fp32 register pressure (the worst dtype at 55.7%) while keeping fp16/bf16 throughput. cuTile — revert to iter-1's simple 128×128 "one scalar scale load" design (don't repeat iter-2's regression) and bump `occupancy=4` to push more programs in flight on a memory-bound kernel.
