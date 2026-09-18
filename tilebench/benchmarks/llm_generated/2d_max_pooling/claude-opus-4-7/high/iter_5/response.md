Looking at the trajectory:
- **Triton**: iter 2 (BLOCK=1024 flat) was best at 36.9%. Variants regressed. Best lever left is L2-cache friendly eviction hints + assume hints, since this is bandwidth-bound with heavy input reuse.
- **cuTile**: iter 1 (BLOCK=512 flat) was best at 15.8%. Will try larger BLOCK with lower occupancy to give each block more register budget and better latency hiding.

Strategy: keep the proven flat 1D output indexing (same as best iter) but add `eviction_policy="evict_last"` on the input loads so overlapping windows hit L2, and add `tl.assume` for address-math folding.

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
    KS: tl.constexpr, BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    pid = tl.program_id(1)

    tl.assume(stride > 0)
    tl.assume(padding >= 0)
    tl.assume(H > 0)
    tl.assume(W > 0)
    tl.assume(H_out > 0)
    tl.assume(W_out > 0)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    n_out = H_out * W_out
    mask_out = offs < n_out

    h_out = offs // W_out
    w_out = offs % W_out

    h_start = h_out * stride - padding
    w_start = w_out * stride - padding

    nc_base = nc * H * W

    NEG_INF = float('-inf')
    acc = tl.full((BLOCK,), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        h = h_start + kh
        valid_h = (h >= 0) & (h < H)
        row_base = nc_base + h * W
        for kw in tl.static_range(KS):
            w = w_start + kw
            valid = (w >= 0) & (w < W) & valid_h & mask_out
            idx = row_base + w
            v = tl.load(
                x_ptr + idx, mask=valid, other=NEG_INF,
                eviction_policy="evict_last",
            )
            acc = tl.maximum(acc, v.to(tl.float32))

    out_off = nc * n_out + offs
    tl.store(out_ptr + out_off, acc, mask=mask_out)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK = 1024
    num_warps = 8
    num_stages = 2

    NC = N * C
    grid = (NC, triton.cdiv(H_out * W_out, BLOCK))

    _max_pool_kernel[grid](
        input, output,
        H, W, H_out, W_out,
        stride, padding,
        KS=kernel_size, BLOCK=BLOCK,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK": BLOCK,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "KS": kernel_size,
        "evict": "evict_last",
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


@ct.kernel(occupancy=2)
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

    BLOCK = 2048
    occupancy = 2

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

Approach: revert to best-known flat 1D layout for both backends. For Triton, add `eviction_policy="evict_last"` on input loads (overlapping pool windows share data across blocks → keep in L2) plus `tl.assume` hints to fold address math. For cuTile, bump BLOCK to 2048 with occupancy=2 to give larger working sets per CTA while maintaining enough resident blocks for latency hiding.
