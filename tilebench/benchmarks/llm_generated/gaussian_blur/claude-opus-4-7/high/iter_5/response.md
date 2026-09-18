Looking at the trajectory, all configs cluster around 5-7% roofline. The fundamental issue is data reuse: each input pixel is reloaded ~49 times across kernel positions. I'll try two different angles:

- **Triton**: Try an untested tile shape (small M × large N) for better row-coalescing with same overall data fit, since all variations cluster near 5-6%.
- **cuTile**: Implement an explicit **halo-load** approach — load the full input region (with padding-to-pow2) once per output tile into a single tile, then use `ct.extract()` to slice sub-tiles for each kernel tap. This dramatically reduces redundant global memory traffic.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _conv2d_kernel(x_ptr, k_ptr, out_ptr,
                   H, W,
                   KH: tl.constexpr, KW: tl.constexpr,
                   PAD_H: tl.constexpr, PAD_W: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            row = offs_m[:, None] + (kh - PAD_H)
            col = offs_n[None, :] + (kw - PAD_W)
            mask = (row >= 0) & (row < H) & (col >= 0) & (col < W)
            ptrs = x_ptr + row * W + col
            x_val = tl.load(ptrs, mask=mask, other=0.0,
                            eviction_policy="evict_last").to(tl.float32)
            w_val = tl.load(k_ptr + kh * KW + kw).to(tl.float32)
            acc += x_val * w_val

    out_mask = (offs_m[:, None] < H) & (offs_n[None, :] < W)
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    BLOCK_M = 8
    BLOCK_N = 512
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(W, BLOCK_N))
    _conv2d_kernel[grid](
        input, kernel, output, H, W,
        KH=KH, KW=KW, PAD_H=PAD_H, PAD_W=PAD_W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
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
def _conv2d_kernel(x, k, out,
                   KH: ConstInt, KW: ConstInt,
                   PAD_H: ConstInt, PAD_W: ConstInt,
                   BM: ConstInt, BN: ConstInt,
                   HM: ConstInt, HN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    base_m = bid_m * BM - PAD_H
    base_n = bid_n * BN - PAD_W

    rows = ct.arange(HM, dtype=np.int32) + base_m       # [HM]
    cols = ct.arange(HN, dtype=np.int32) + base_n       # [HN]

    row2d = ct.broadcast_to(rows[:, None], (HM, HN))
    col2d = ct.broadcast_to(cols[None, :], (HM, HN))

    # Load halo'd input region once
    halo = ct.gather(x, (row2d, col2d), padding_value=0.0)
    halo_f32 = ct.astype(halo, np.float32)

    acc = ct.full((BM, BN), 0.0, dtype=np.float32)

    # Reuse halo for all kernel taps via static sub-tile extraction
    for kh in range(KH):
        for kw in range(KW):
            sub = ct.extract(halo_f32, (kh, kw), shape=(BM, BN))
            w_val = ct.load(k, index=(kh, kw), shape=())
            acc = acc + sub * ct.astype(w_val, np.float32)

    ct.store(out, index=(bid_m, bid_n), tile=ct.astype(acc, x.dtype))


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    x2d = input.view(H, W)
    k2d = kernel.view(KH, KW)
    out2d = output.view(H, W)

    # Output tile (must be pow2)
    BM = 32
    BN = 64
    # Halo tile (must be pow2, >= BM+KH-1 and BN+KW-1).
    # For KH=KW=7: BM+6=38 -> HM=64; BN+6=70 -> HN=128.
    HM = 64
    HN = 128
    occupancy = 2

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(H, BM), ct.cdiv(W, BN), 1)
    ct.launch(stream, grid, _conv2d_kernel,
              (x2d, k2d, out2d, KH, KW, PAD_H, PAD_W, BM, BN, HM, HN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BM": BM,
        "BN": BN,
        "HM": HM,
        "HN": HN,
        "occupancy": occupancy,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
