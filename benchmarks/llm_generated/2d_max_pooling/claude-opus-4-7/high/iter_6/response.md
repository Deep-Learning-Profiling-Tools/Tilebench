Looking at the trajectory, Triton has plateaued around 36-37% with 1D BLOCK approaches. The previous 2D attempt (iter 1) used BLOCK_W=64 which split rows across programs, hurting locality. Let me try a 2D layout where BLOCK_W ≥ W_out (=320 for the largest case) so each program handles full output rows — this keeps loads contiguous along W and avoids row-crossing within a program.

For cuTile, all attempts regressed vs iter 1's BLOCK=512 occupancy=4 — reverting to that.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _max_pool_kernel(
    x_ptr, out_ptr,
    H, W, H_out, W_out,
    stride, padding,
    KS: tl.constexpr,
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    nc = tl.program_id(0)
    ph = tl.program_id(1)

    tl.assume(stride > 0)
    tl.assume(padding >= 0)
    tl.assume(H > 0)
    tl.assume(W > 0)
    tl.assume(H_out > 0)
    tl.assume(W_out > 0)

    h_out = ph * BLOCK_H + tl.arange(0, BLOCK_H)        # [BLOCK_H]
    w_out = tl.arange(0, BLOCK_W)                       # [BLOCK_W]
    mask_h = h_out < H_out
    mask_w = w_out < W_out

    h_in = h_out * stride - padding                     # [BLOCK_H]
    w_in = w_out * stride - padding                     # [BLOCK_W]

    nc_base = nc * H * W
    NEG_INF = float('-inf')
    acc = tl.full((BLOCK_H, BLOCK_W), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        h = h_in[:, None] + kh                          # [BLOCK_H, 1]
        valid_h = (h >= 0) & (h < H)
        for kw in tl.static_range(KS):
            w = w_in[None, :] + kw                      # [1, BLOCK_W]
            valid_w = (w >= 0) & (w < W)
            valid = valid_h & valid_w & mask_h[:, None] & mask_w[None, :]
            idx = nc_base + h * W + w
            v = tl.load(x_ptr + idx, mask=valid, other=NEG_INF)
            acc = tl.maximum(acc, v.to(tl.float32))

    out_idx = nc * H_out * W_out + h_out[:, None] * W_out + w_out[None, :]
    mask_out = mask_h[:, None] & mask_w[None, :]
    tl.store(
        out_ptr + out_idx,
        acc.to(out_ptr.dtype.element_ty),
        mask=mask_out,
    )


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    # Choose BLOCK_W as smallest pow-2 >= W_out so each program holds full output rows
    # (avoids row-crossing within a tile -> coalesced loads along W).
    BLOCK_W = max(64, triton.next_power_of_2(W_out))
    BLOCK_H = 4
    num_warps = 8
    num_stages = 2

    NC = N * C
    grid = (NC, triton.cdiv(H_out, BLOCK_H))

    _max_pool_kernel[grid](
        input, output,
        H, W, H_out, W_out,
        stride, padding,
        KS=kernel_size,
        BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_H": BLOCK_H,
        "BLOCK_W": BLOCK_W,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "KS": kernel_size,
    })
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


@ct.kernel(occupancy=4)
def _max_pool_kernel(
    x_flat, out_flat,
    H, W, H_out, W_out,
    stride, padding,
    KS: ConstInt, BLOCK: ConstInt,
):
    nc = ct.bid(0)
    pid = ct.bid(1)

    offs = pid * BLOCK + ct.arange(BLOCK, dtype=np.int32)
    n_out = H_out * W_out
    valid_out = offs < n_out

    h_out = offs // W_out
    w_out = offs % W_out

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    nc_base = nc * H * W

    acc = ct.full((BLOCK,), -np.inf, dtype=np.float32)
    neg_inf_tile = ct.full((BLOCK,), -np.inf, dtype=np.float32)

    for kh in range(KS):
        for kw in range(KS):
            h = h_start + kh
            w = w_start + kw
            valid = (h >= 0) & (h < H) & (w >= 0) & (w < W) & valid_out
            idx = nc_base + h * W + w
            safe_idx = ct.where(valid, idx, -1)
            v = ct.gather(x_flat, safe_idx, padding_value=0)
            vf = ct.astype(v, np.float32)
            vf = ct.where(valid, vf, neg_inf_tile)
            acc = ct.maximum(acc, vf)

    out_idx = nc * n_out + offs
    safe_out = ct.where(valid_out, out_idx, -1)
    ct.scatter(out_flat, safe_out, ct.astype(acc, x_flat.dtype))


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK = 512
    occupancy = 4

    NC = N * C
    grid = (NC, ct.cdiv(H_out * W_out, BLOCK), 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, grid, _max_pool_kernel,
        (input, output, H, W, H_out, W_out, stride, padding, kernel_size, BLOCK),
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK,
        "occupancy": occupancy,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
