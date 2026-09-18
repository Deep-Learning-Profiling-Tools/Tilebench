import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < n_cols
    offsets = row * n_cols + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)

    numerator = tl.exp2(x * 1.4426950408889634)
    denominator = tl.sum(numerator, axis=0)
    y = numerator / denominator

    tl.store(out_ptr + offsets, y, mask=mask)


def run(x):
    output = torch.empty_like(x)
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    BLOCK_N = triton.next_power_of_2(n_cols)

    num_warps = 4
    if BLOCK_N >= 2048:
        num_warps = 8
    if BLOCK_N >= 8192:
        num_warps = 16
    num_stages = 4

    grid = (n_rows,)
    _softmax_kernel[grid](
        x,
        output,
        n_cols,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update(
        {
            "BLOCK_N": BLOCK_N,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    )
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
