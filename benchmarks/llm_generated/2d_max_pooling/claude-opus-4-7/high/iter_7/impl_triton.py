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
    BLOCK_W: tl.constexpr,
    BLOCK_WIN: tl.constexpr,
):
    nc = tl.program_id(0)
    h_out = tl.program_id(1)
    w_blk = tl.program_id(2)

    tl.assume(stride > 0)
    tl.assume(padding >= 0)
    tl.assume(H > 0)
    tl.assume(W > 0)

    nc_base = nc * H * W
    h_in_base = h_out * stride - padding
    w_out_base = w_blk * BLOCK_W
    w_in_base = w_out_base * stride - padding

    NEG_INF = float('-inf')

    # Contiguous input-column strip — no stride-2 cache-line waste
    win_offs = tl.arange(0, BLOCK_WIN)
    w_in = w_in_base + win_offs
    win_limit = BLOCK_W * stride + KS - 1
    valid_win = (w_in >= 0) & (w_in < W) & (win_offs < win_limit)

    w_out_range = tl.arange(0, BLOCK_W)
    w_out_full = w_out_base + w_out_range
    mask_w_out = w_out_full < W_out

    acc = tl.full((BLOCK_W,), NEG_INF, dtype=tl.float32)

    for kh in tl.static_range(KS):
        h_in = h_in_base + kh
        valid_h = (h_in >= 0) & (h_in < H)
        mask_load = valid_h & valid_win
        idx = nc_base + h_in * W + w_in
        row = tl.load(x_ptr + idx, mask=mask_load, other=NEG_INF)
        rowf = row.to(tl.float32)
        # Pick strided sub-positions from registers (no extra DRAM traffic)
        for kw in tl.static_range(KS):
            gather_idx = w_out_range * stride + kw
            picked = tl.gather(rowf, gather_idx, axis=0)
            acc = tl.maximum(acc, picked)

    out_off = nc * H_out * W_out + h_out * W_out + w_out_full
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=mask_w_out)


def run(input, N, C, H, W, kernel_size, stride, padding, **kwargs):
    H_out = (H + 2 * padding - kernel_size) // stride + 1
    W_out = (W + 2 * padding - kernel_size) // stride + 1
    output = torch.empty(N * C * H_out * W_out, dtype=input.dtype, device=input.device)

    BLOCK_W = 256
    # max gather idx = (BLOCK_W - 1) * stride + KS - 1; need BLOCK_WIN > that
    BLOCK_WIN = triton.next_power_of_2(BLOCK_W * stride + kernel_size - 1)

    num_warps = 4
    num_stages = 2

    NC = N * C
    grid = (NC, H_out, triton.cdiv(W_out, BLOCK_W))

    _max_pool_kernel[grid](
        input, output,
        H, W, H_out, W_out,
        stride, padding,
        KS=kernel_size,
        BLOCK_W=BLOCK_W,
        BLOCK_WIN=BLOCK_WIN,
        num_warps=num_warps, num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_W": BLOCK_W,
        "BLOCK_WIN": BLOCK_WIN,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "KS": kernel_size,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
