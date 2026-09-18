Looking at the data: we're at ~30% roofline on fp16/bf16 (bandwidth-bound). Best case is ~5 reads/element with no L2 reuse, so we need better L2 reuse to get higher. I'll apply matmul-style L2 swizzle (GROUP_M) for Triton to keep neighboring rows hot in L2, and try a smaller-M / wider-N tile for cuTile to increase L2 sharing across adjacent CTAs.

```python title="impl_triton.py"
import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _jacobi_kernel(in_ptr, out_ptr, rows, cols,
                   stride_r, stride_c,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                   GROUP_M: tl.constexpr):
    # Matmul-style program-ID swizzle for L2 reuse.
    # Stencil neighbour rows (up/down) are loaded by adjacent-M CTAs;
    # grouping CTAs in M keeps those rows hot in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(cols, BLOCK_N)
    num_in_group = GROUP_M * num_pid_n
    group_id = pid // num_in_group
    first_pid_m = group_id * GROUP_M
    gsize = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_in_group) % gsize)
    pid_n = (pid % num_in_group) // gsize

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(stride_r > 0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rm = offs_m[:, None]
    cn = offs_n[None, :]

    in_bounds = (rm < rows) & (cn < cols)
    interior = (rm >= 1) & (rm < rows - 1) & (cn >= 1) & (cn < cols - 1)

    base = rm * stride_r + cn * stride_c

    center = tl.load(in_ptr + base, mask=in_bounds, other=0.0,
                     eviction_policy="evict_first").to(tl.float32)

    up = tl.load(in_ptr + (rm - 1) * stride_r + cn * stride_c,
                 mask=interior, other=0.0,
                 eviction_policy="evict_last").to(tl.float32)
    down = tl.load(in_ptr + (rm + 1) * stride_r + cn * stride_c,
                   mask=interior, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    left = tl.load(in_ptr + rm * stride_r + (cn - 1) * stride_c,
                   mask=interior, other=0.0,
                   eviction_policy="evict_last").to(tl.float32)
    right = tl.load(in_ptr + rm * stride_r + (cn + 1) * stride_c,
                    mask=interior, other=0.0,
                    eviction_policy="evict_last").to(tl.float32)

    avg = 0.25 * (up + down + left + right)
    result = tl.where(interior, avg, center)

    tl.store(out_ptr + base, result, mask=in_bounds)


def run(input: torch.Tensor, rows: int, cols: int, **kwargs):
    output = torch.empty_like(input)

    BLOCK_M = 8
    BLOCK_N = 256
    GROUP_M = 8
    num_warps = 4
    num_stages = 2

    num_pid_m = triton.cdiv(rows, BLOCK_M)
    num_pid_n = triton.cdiv(cols, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    _jacobi_kernel[grid](
        input, output, rows, cols,
        input.stride(0), input.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "GROUP_M": GROUP_M,
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

    # Smaller M, wider N: each CTA's neighbour-row footprint is small,
    # so adjacent-M CTAs share more lines in L2.
    BM = 4
    BN = 512

    grid = (ct.cdiv(rows, BM), ct.cdiv(cols, BN), 1)
    ct.launch(stream, grid, _jacobi_kernel,
              (input, output, rows, cols, BM, BN))

    _LAST_CFG.clear()
    _LAST_CFG.update({"BM": BM, "BN": BN, "occupancy": 2})
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
```

Approach: For Triton, added matmul-style GROUP_M=8 program-ID swizzle to keep neighbor rows hot in L2 (stencil neighbors are loaded by adjacent-M CTAs). For cuTile, shrunk M to 4 and widened N to 512 to maximize cross-CTA L2 reuse of vertical neighbor rows.
