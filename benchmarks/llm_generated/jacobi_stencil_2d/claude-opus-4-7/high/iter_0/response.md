Looking at this Jacobi 5-point stencil:
- **Triton**: 2D tile kernel, one tl.load for center + 4 shifted tl.loads for neighbors (L1 cache catches reuse), select interior-vs-boundary via mask.
- **cuTile**: aligned ct.load for center + ct.gather for the 4 shifted neighbors (tile-space indexing can't easily express ±1 element offsets), then aligned ct.store.

Compute always in fp32 (matches reference's `input.float()`) and cast back via store/explicit astype.

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

    base = rm * stride_r + cn * stride_c
    center = tl.load(in_ptr + base, mask=in_bounds, other=0.0).to(tl.float32)

    up = tl.load(in_ptr + (rm - 1) * stride_r + cn * stride_c,
                 mask=interior, other=0.0).to(tl.float32)
    down = tl.load(in_ptr + (rm + 1) * stride_r + cn * stride_c,
                   mask=interior, other=0.0).to(tl.float32)
    left = tl.load(in_ptr + rm * stride_r + (cn - 1) * stride_c,
                   mask=interior, other=0.0).to(tl.float32)
    right = tl.load(in_ptr + rm * stride_r + (cn + 1) * stride_c,
                    mask=interior, other=0.0).to(tl.float32)

    avg = 0.25 * (up + down + left + right)
    result = tl.where(interior, avg, center)

    tl.store(out_ptr + base, result, mask=in_bounds)


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_M = 16
    BLOCK_N = 128
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


@ct.kernel
def _jacobi_kernel(input, output,
                   rows: ConstInt, cols: ConstInt,
                   BM: ConstInt, BN: ConstInt):
    bm = ct.bid(0)
    bn = ct.bid(1)

    rs = ct.arange(BM, dtype=np.int32) + bm * BM   # [BM]
    cs = ct.arange(BN, dtype=np.int32) + bn * BN   # [BN]

    r_2d = ct.broadcast_to(rs[:, None], (BM, BN))
    c_2d = ct.broadcast_to(cs[None, :], (BM, BN))

    center = ct.load(input, index=(bm, bn), shape=(BM, BN),
                     padding_mode=ct.PaddingMode.ZERO)

    up    = ct.gather(input, (r_2d - 1, c_2d), padding_value=0)
    down  = ct.gather(input, (r_2d + 1, c_2d), padding_value=0)
    left  = ct.gather(input, (r_2d, c_2d - 1), padding_value=0)
    right = ct.gather(input, (r_2d, c_2d + 1), padding_value=0)

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

    BM = 32
    BN = 64
    occupancy = 2

    grid = (ct.cdiv(rows, BM), ct.cdiv(cols, BN), 1)
    kernel = _jacobi_kernel.with_hints(occupancy=occupancy)
    ct.launch(stream, grid, kernel,
              (input, output, rows, cols, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BM": BM, "BN": BN, "occupancy": occupancy})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: each program processes one (BM × BN) output tile. The center load + tl.where mask select interior-vs-boundary; only interior cells average the 4 cardinal neighbors (loaded with mask=interior so boundary tiles avoid unnecessary OOB reads). cuTile uses gather since tile-space indexing can't express ±1-element shifts directly.
