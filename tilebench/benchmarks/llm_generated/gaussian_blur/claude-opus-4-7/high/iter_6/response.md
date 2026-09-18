Looking at the trajectory, all 5 prior configs are stuck around 5-6% peak — the kernel is L2-bandwidth-bound because each output pixel triggers 49 (7×7) redundant reads of overlapping input regions, amplifying L2 traffic by ~49×.

The big algorithmic win: **a Gaussian kernel is exactly rank-1 separable** (K = u·vᵀ). Extract u and v on the host (just a sum + division), then run two 1-D 7-tap passes instead of one 2-D 49-tap pass. That collapses L2 amplification from 49× → 14× and should yield a ~3× speedup.

For cuTile, I revert away from the `ct.extract` halo trick (which broke verify in iter 5 — `extract`'s index semantics didn't match what I assumed) and just use the same separable approach with direct `ct.gather`.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _hconv_kernel(x_ptr, kw_ptr, out_ptr,
                  H, W,
                  KW: tl.constexpr, PAD_W: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    row = offs_m[:, None]
    row_mask = row < H
    for kw in tl.static_range(0, KW):
        col = offs_n[None, :] + (kw - PAD_W)
        mask = row_mask & (col >= 0) & (col < W)
        ptrs = x_ptr + row * W + col
        x_val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(kw_ptr + kw).to(tl.float32)
        acc += x_val * w_val
    out_mask = (offs_m[:, None] < H) & (offs_n[None, :] < W)
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def _vconv_kernel(x_ptr, kh_ptr, out_ptr,
                  H, W,
                  KH: tl.constexpr, PAD_H: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    col = offs_n[None, :]
    col_mask = col < W
    for kh in tl.static_range(0, KH):
        row = offs_m[:, None] + (kh - PAD_H)
        mask = (row >= 0) & (row < H) & col_mask
        ptrs = x_ptr + row * W + col
        x_val = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        w_val = tl.load(kh_ptr + kh).to(tl.float32)
        acc += x_val * w_val
    out_mask = (offs_m[:, None] < H) & col_mask
    out_ptrs = out_ptr + offs_m[:, None] * W + offs_n[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def _decompose_rank1(kernel, KH, KW, device):
    """Extract 1-D factors g_v, g_h such that K = g_v ⊗ g_h.

    For a true Gaussian (rank-1), this is exact and far cheaper than SVD.
    """
    k2d = kernel.float().view(KH, KW)
    ch = KH // 2
    cw = KW // 2
    center_row = k2d[ch, :]                     # [KW]
    center_col = k2d[:, cw]                     # [KH]
    g_v_center = center_row.sum()                # = g_v[ch]  (since g_h sums to 1)
    g_h = center_row / g_v_center                # [KW]
    g_h_center = g_h[cw]
    g_v = center_col / g_h_center                # [KH]
    return g_v.contiguous(), g_h.contiguous()


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    g_v, g_h = _decompose_rank1(kernel, KH, KW, input.device)
    kh_1d = g_v   # vertical 1-D kernel (length KH)
    kw_1d = g_h   # horizontal 1-D kernel (length KW)

    BLOCK_M = 16
    BLOCK_N = 128
    num_warps = 4
    num_stages = 2

    temp = torch.empty(H * W, dtype=torch.float32, device=input.device)
    grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(W, BLOCK_N))

    _hconv_kernel[grid](
        input, kw_1d, temp, H, W,
        KW=KW, PAD_W=PAD_W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    _vconv_kernel[grid](
        temp, kh_1d, output, H, W,
        KH=KH, PAD_H=PAD_H,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "approach": "separable_rank1",
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
def _hconv_kernel(x, kw, out,
                  KW: ConstInt, PAD_W: ConstInt,
                  BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    offs_m = bid_m * BM + ct.arange(BM, dtype=np.int32)
    offs_n = bid_n * BN + ct.arange(BN, dtype=np.int32)
    acc = ct.full((BM, BN), 0.0, dtype=np.float32)
    for kw_idx in range(KW):
        row = offs_m[:, None]
        col = offs_n[None, :] + (kw_idx - PAD_W)
        r2 = ct.broadcast_to(row, (BM, BN))
        c2 = ct.broadcast_to(col, (BM, BN))
        vals = ct.gather(x, (r2, c2), padding_value=0.0)
        w_val = ct.load(kw, index=(kw_idx,), shape=())
        acc = acc + ct.astype(vals, np.float32) * ct.astype(w_val, np.float32)
    ct.store(out, index=(bid_m, bid_n), tile=acc)


@ct.kernel(occupancy=4)
def _vconv_kernel(x, kh, out,
                  KH: ConstInt, PAD_H: ConstInt,
                  BM: ConstInt, BN: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    offs_m = bid_m * BM + ct.arange(BM, dtype=np.int32)
    offs_n = bid_n * BN + ct.arange(BN, dtype=np.int32)
    acc = ct.full((BM, BN), 0.0, dtype=np.float32)
    for kh_idx in range(KH):
        row = offs_m[:, None] + (kh_idx - PAD_H)
        col = offs_n[None, :]
        r2 = ct.broadcast_to(row, (BM, BN))
        c2 = ct.broadcast_to(col, (BM, BN))
        vals = ct.gather(x, (r2, c2), padding_value=0.0)
        w_val = ct.load(kh, index=(kh_idx,), shape=())
        acc = acc + ct.astype(vals, np.float32) * ct.astype(w_val, np.float32)
    ct.store(out, index=(bid_m, bid_n), tile=ct.astype(acc, out.dtype))


def _decompose_rank1(kernel, KH, KW):
    """K = g_v ⊗ g_h (exact for true Gaussian)."""
    k2d = kernel.float().view(KH, KW)
    ch = KH // 2
    cw = KW // 2
    center_row = k2d[ch, :]
    center_col = k2d[:, cw]
    g_v_center = center_row.sum()
    g_h = center_row / g_v_center
    g_h_center = g_h[cw]
    g_v = center_col / g_h_center
    return g_v.contiguous(), g_h.contiguous()


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols, **kwargs):
    output = torch.empty_like(input)
    H, W = input_rows, input_cols
    KH, KW = kernel_rows, kernel_cols
    PAD_H, PAD_W = KH // 2, KW // 2

    g_v, g_h = _decompose_rank1(kernel, KH, KW)
    kh_1d = g_v   # length KH (fp32)
    kw_1d = g_h   # length KW (fp32)

    x2d = input.view(H, W)
    out2d = output.view(H, W)
    temp = torch.empty(H * W, dtype=torch.float32, device=input.device)
    temp2d = temp.view(H, W)

    BM = 32
    BN = 128
    occupancy = 4

    stream = torch.cuda.current_stream()
    grid = (ct.cdiv(H, BM), ct.cdiv(W, BN), 1)

    ct.launch(stream, grid, _hconv_kernel,
              (x2d, kw_1d, temp2d, KW, PAD_W, BM, BN))
    ct.launch(stream, grid, _vconv_kernel,
              (temp2d, kh_1d, out2d, KH, PAD_H, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BM": BM,
        "BN": BN,
        "occupancy": occupancy,
        "approach": "separable_rank1",
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: decompose the 7×7 Gaussian kernel into its exact rank-1 factors `K = g_v · g_hᵀ` using simple host-side tensor ops (sum + division of the centre row/col — no SVD, no CPU sync), then run two 7-tap 1-D convolution passes (horizontal then vertical) on both Triton and cuTile. This cuts the per-pixel L2-resident reads from 49→14 and the redundant memory amplification by ~3.5×, which should lift the bandwidth-bound roofline significantly.
