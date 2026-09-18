import torch
import triton
import triton.language as tl

_LAST_CFG: dict = {}


@triton.jit
def _softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                    n_rows, n_cols,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols,
                mask=mask, other=-float('inf')).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    y = num / den
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def run(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    if BLOCK_SIZE >= 16384:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 512:
        num_warps = 4
    else:
        num_warps = 2

    num_stages = 2

    grid = (n_rows,)
    _softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),
        n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    _LAST_CFG.clear()
    _LAST_CFG.update({
        "BLOCK_SIZE": BLOCK_SIZE,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "num_programs": n_rows,
    })
    return output


def get_last_config() -> dict | None:
    return dict(_LAST_CFG) if _LAST_CFG else None
