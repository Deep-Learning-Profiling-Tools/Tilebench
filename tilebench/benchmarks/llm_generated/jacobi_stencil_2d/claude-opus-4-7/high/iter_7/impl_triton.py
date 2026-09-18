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
