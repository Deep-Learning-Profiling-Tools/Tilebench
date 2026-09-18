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
