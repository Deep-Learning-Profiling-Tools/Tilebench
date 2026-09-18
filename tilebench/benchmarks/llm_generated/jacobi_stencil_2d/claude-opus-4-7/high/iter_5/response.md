Looking at iteration 4's results: stable around 30-38% roofline. The kernels are bandwidth-bound, but the bottleneck is that each load carries a 4-condition mask and we have boundary-handling overhead.

**Optimization for iter 5**: Eliminate per-load masks entirely by **clamping neighbor indices** to `[0, rows-1] × [0, cols-1]`. This makes all loads guaranteed in-bounds, allowing:
- Triton: drop the `mask=interior` on the 4 neighbor loads (and `mask=in_bounds` on center) — only the final `tl.store` needs a mask
- cuTile: pass `check_bounds=False` to `ct.gather` — eliminates per-element bounds checks in the gather lowering

The result is semantically identical (interior cells always use the 4 neighbors, boundary cells get overwritten by `tl.where(interior, avg, center)` to the un-shifted center value). We keep iter 4's best block shapes.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_kernel(in_ptr, out_ptr, rows, cols,
                   stride_r, stride_c,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm = offs_m[:, None]
    cn = offs_n[None, :]

    in_bounds = (rm < rows) & (cn < cols)
    interior = (rm >= 1) & (rm < rows - 1) & (cn >= 1) & (cn < cols - 1)

    # Clamp every index to the valid array range so EVERY load is safe
    # without a per-load mask. The final tl.store still uses in_bounds.
    rm_c = tl.minimum(rm, rows - 1)
    cn_c = tl.minimum(cn, cols - 1)
    rm_up = tl.maximum(rm_c - 1, 0)
    rm_dn = tl.minimum(rm_c + 1, rows - 1)
    cn_lf = tl.maximum(cn_c - 1, 0)
    cn_rt = tl.minimum(cn_c + 1, cols - 1)

    # 5 unmasked loads — guaranteed in-bounds via clamping above
    center = tl.load(in_ptr + rm_c * stride_r + cn_c * stride_c,
                     eviction_policy="evict_last").to(tl.float32)
    up = tl.load(in_ptr + rm_up * stride_r + cn_c * stride_c,
                 eviction_policy="evict_last").to(tl.float32)
    down = tl.load(in_ptr + rm_dn * stride_r + cn_c * stride_c,
                   eviction_policy="evict_last").to(tl.float32)
    left = tl.load(in_ptr + rm_c * stride_r + cn_lf * stride_c,
                   eviction_policy="evict_last").to(tl.float32)
    right = tl.load(in_ptr + rm_c * stride_r + cn_rt * stride_c,
                    eviction_policy="evict_last").to(tl.float32)

    avg = 0.25 * (up + down + left + right)
    result = tl.where(interior, avg, center)

    tl.store(out_ptr + rm * stride_r + cn * stride_c, result, mask=in_bounds)


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_M = 8
    BLOCK_N = 256
    num_warps = 4
    num_stages = 2

    grid = (triton.cdiv(rows, BLOCK_M), triton.cdiv(cols, BLOCK_N))
    _jacobi_kernel[grid](
        input, output, rows, cols,
        input.stride(0), input.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
        "num_warps": num_warps, "num_stages": num_stages,
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
def _jacobi_kernel(input, output, rows, cols,
                   BM: ConstInt, BN: ConstInt):
    bm = ct.bid(0)
    bn = ct.bid(1)

    rs = ct.arange(BM, dtype=np.int32) + bm * BM
    cs = ct.arange(BN, dtype=np.int32) + bn * BN

    r_2d = ct.broadcast_to(rs[:, None], (BM, BN))
    c_2d = ct.broadcast_to(cs[None, :], (BM, BN))

    # Clamp so every gather index is guaranteed in-bounds → skip bounds-check.
    r_c = ct.minimum(r_2d, rows - 1)
    c_c = ct.minimum(c_2d, cols - 1)
    r_up = ct.maximum(r_c - 1, 0)
    r_dn = ct.minimum(r_c + 1, rows - 1)
    c_lf = ct.maximum(c_c - 1, 0)
    c_rt = ct.minimum(c_c + 1, cols - 1)

    center = ct.gather(input, (r_c, c_c), check_bounds=False)
    up    = ct.gather(input, (r_up, c_c), check_bounds=False)
    down  = ct.gather(input, (r_dn, c_c), check_bounds=False)
    left  = ct.gather(input, (r_c, c_lf), check_bounds=False)
    right = ct.gather(input, (r_c, c_rt), check_bounds=False)

    center_f = ct.astype(center, np.float32)
    up_f     = ct.astype(up,     np.float32)
    down_f   = ct.astype(down,   np.float32)
    left_f   = ct.astype(left,   np.float32)
    right_f  = ct.astype(right,  np.float32)

    avg = 0.25 * (up_f + down_f + left_f + right_f)

    is_interior = (r_2d > 0) & (r_2d < rows - 1) & (c_2d > 0) & (c_2d < cols - 1)
    result_f = ct.where(is_interior, avg, center_f)

    result = ct.astype(result_f, input.dtype)
    ct.store(output, index=(bm, bn), tile=result)


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)
    stream = torch.cuda.current_stream()

    BM = 16
    BN = 128

    grid = (ct.cdiv(rows, BM), ct.cdiv(cols, BN), 1)
    ct.launch(stream, grid, _jacobi_kernel,
              (input, output, rows, cols, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BM": BM, "BN": BN, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```
