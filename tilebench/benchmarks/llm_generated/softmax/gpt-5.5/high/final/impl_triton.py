import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _softmax_kernel_pow2(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    base = row * n_cols

    x = tl.load(x_ptr + base + cols, mask=mask, other=-float("inf"))
    row_max = tl.max(x, axis=0).to(tl.float32)

    p = tl.exp2((x.to(tl.float32) - row_max) * 1.4426950408889634).to(x.dtype)
    denominator = tl.sum(p.to(tl.float32), axis=0)
    inv_denominator = 1.0 / denominator

    tl.store(out_ptr + base + cols, p.to(tl.float32) * inv_denominator, mask=mask)


@triton.jit
def _softmax_kernel_split5_nomask(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    base = row * n_cols

    c0 = offs
    c1 = offs + BLOCK_N
    c2 = offs + 2 * BLOCK_N
    c3 = offs + 3 * BLOCK_N
    c4 = offs + 4 * BLOCK_N

    x0 = tl.load(x_ptr + base + c0, eviction_policy="evict_first")
    x1 = tl.load(x_ptr + base + c1, eviction_policy="evict_first")
    x2 = tl.load(x_ptr + base + c2, eviction_policy="evict_first")
    x3 = tl.load(x_ptr + base + c3, eviction_policy="evict_first")
    x4 = tl.load(x_ptr + base + c4, eviction_policy="evict_first")

    xmax01 = tl.maximum(x0, x1)
    xmax23 = tl.maximum(x2, x3)
    xmax = tl.maximum(tl.maximum(xmax01, xmax23), x4)
    row_max = tl.max(xmax, axis=0).to(tl.float32)

    # For fp16 inputs, keep numerators in fp16.  The final normalization is
    # still fp32, but this sharply reduces live fp32 vectors in the large-row
    # kernel.  For fp32 inputs this is a no-op and preserves the exact path.
    p0 = tl.exp2((x0.to(tl.float32) - row_max) * 1.4426950408889634).to(x0.dtype)
    p1 = tl.exp2((x1.to(tl.float32) - row_max) * 1.4426950408889634).to(x1.dtype)
    p2 = tl.exp2((x2.to(tl.float32) - row_max) * 1.4426950408889634).to(x2.dtype)
    p3 = tl.exp2((x3.to(tl.float32) - row_max) * 1.4426950408889634).to(x3.dtype)
    p4 = tl.exp2((x4.to(tl.float32) - row_max) * 1.4426950408889634).to(x4.dtype)

    denominator = tl.sum((p0 + p1 + p2 + p3 + p4).to(tl.float32), axis=0)
    inv_denominator = 1.0 / denominator

    tl.store(out_ptr + base + c0, p0.to(tl.float32) * inv_denominator, cache_modifier=".cs")
    tl.store(out_ptr + base + c1, p1.to(tl.float32) * inv_denominator, cache_modifier=".cs")
    tl.store(out_ptr + base + c2, p2.to(tl.float32) * inv_denominator, cache_modifier=".cs")
    tl.store(out_ptr + base + c3, p3.to(tl.float32) * inv_denominator, cache_modifier=".cs")
    tl.store(out_ptr + base + c4, p4.to(tl.float32) * inv_denominator, cache_modifier=".cs")


@triton.jit
def _softmax_kernel_split5_masked(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    base = row * n_cols

    c0 = offs
    c1 = offs + BLOCK_N
    c2 = offs + 2 * BLOCK_N
    c3 = offs + 3 * BLOCK_N
    c4 = offs + 4 * BLOCK_N

    m0 = c0 < n_cols
    m1 = c1 < n_cols
    m2 = c2 < n_cols
    m3 = c3 < n_cols
    m4 = c4 < n_cols

    x0 = tl.load(x_ptr + base + c0, mask=m0, other=-float("inf"), eviction_policy="evict_first")
    x1 = tl.load(x_ptr + base + c1, mask=m1, other=-float("inf"), eviction_policy="evict_first")
    x2 = tl.load(x_ptr + base + c2, mask=m2, other=-float("inf"), eviction_policy="evict_first")
    x3 = tl.load(x_ptr + base + c3, mask=m3, other=-float("inf"), eviction_policy="evict_first")
    x4 = tl.load(x_ptr + base + c4, mask=m4, other=-float("inf"), eviction_policy="evict_first")

    xmax01 = tl.maximum(x0, x1)
    xmax23 = tl.maximum(x2, x3)
    xmax = tl.maximum(tl.maximum(xmax01, xmax23), x4)
    row_max = tl.max(xmax, axis=0).to(tl.float32)

    p0 = tl.exp2((x0.to(tl.float32) - row_max) * 1.4426950408889634).to(x0.dtype)
    p1 = tl.exp2((x1.to(tl.float32) - row_max) * 1.4426950408889634).to(x1.dtype)
    p2 = tl.exp2((x2.to(tl.float32) - row_max) * 1.4426950408889634).to(x2.dtype)
    p3 = tl.exp2((x3.to(tl.float32) - row_max) * 1.4426950408889634).to(x3.dtype)
    p4 = tl.exp2((x4.to(tl.float32) - row_max) * 1.4426950408889634).to(x4.dtype)

    denominator = tl.sum((p0 + p1 + p2 + p3 + p4).to(tl.float32), axis=0)
    inv_denominator = 1.0 / denominator

    tl.store(out_ptr + base + c0, p0.to(tl.float32) * inv_denominator, mask=m0, cache_modifier=".cs")
    tl.store(out_ptr + base + c1, p1.to(tl.float32) * inv_denominator, mask=m1, cache_modifier=".cs")
    tl.store(out_ptr + base + c2, p2.to(tl.float32) * inv_denominator, mask=m2, cache_modifier=".cs")
    tl.store(out_ptr + base + c3, p3.to(tl.float32) * inv_denominator, mask=m3, cache_modifier=".cs")
    tl.store(out_ptr + base + c4, p4.to(tl.float32) * inv_denominator, mask=m4, cache_modifier=".cs")


def run(x):
    output = torch.empty_like(x)
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    num_stages = 4

    if n_cols == 10240:
        BLOCK_N = 2048
        num_warps = 16
        grid = (n_rows,)
        _softmax_kernel_split5_nomask[grid](
            x,
            output,
            n_cols,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        cfg = {
            "BLOCK_N": BLOCK_N,
            "N_CHUNKS": 5,
            "TOTAL_BLOCK_N": BLOCK_N * 5,
            "MASKED": 0,
            "MODE": 5,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    elif n_cols > 8192:
        BLOCK_N = 2048
        num_warps = 16
        grid = (n_rows,)
        _softmax_kernel_split5_masked[grid](
            x,
            output,
            n_cols,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        cfg = {
            "BLOCK_N": BLOCK_N,
            "N_CHUNKS": 5,
            "TOTAL_BLOCK_N": BLOCK_N * 5,
            "MASKED": 1,
            "MODE": 5,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    else:
        BLOCK_N = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16

        grid = (n_rows,)
        _softmax_kernel_pow2[grid](
            x,
            output,
            n_cols,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        cfg = {
            "BLOCK_N": BLOCK_N,
            "N_CHUNKS": 1,
            "TOTAL_BLOCK_N": BLOCK_N,
            "MASKED": 1,
            "MODE": 5,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }

    _LAST_CFG.clear()
    _LAST_CFG.update(cfg)
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
