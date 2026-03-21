import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def _swiglu_kernel(
    x_ptr, y_ptr, out_ptr,
    stride_x_row, stride_y_row, stride_out_row,
    ncols,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE

    x_ptr = x_ptr + row * stride_x_row
    y_ptr = y_ptr + row * stride_y_row
    out_ptr = out_ptr + row * stride_out_row

    cols = col_start + tl.arange(0, BLOCK_SIZE)
    mask = cols < ncols

    x = tl.load(x_ptr + cols, mask=mask, other=0.)
    y = tl.load(y_ptr + cols, mask=mask, other=0.)
    out = x * tl.sigmoid(x.to(tl.float32)).to(x.dtype) * y
    tl.store(out_ptr + cols, out, mask=mask)


_swiglu_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [64, 128, 256, 512, 1024, 2048]
        for nw in [4, 8, 16]
    ],
    key=["ncols"],
)(_swiglu_kernel)


def run(x: torch.Tensor, y: torch.Tensor,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    assert x.shape == y.shape
    x = x.contiguous()
    y = y.contiguous()
    M, N = x.shape
    output = torch.empty_like(x)

    if autotune:
        grid = lambda meta: (M, triton.cdiv(N, meta["BLOCK_SIZE"]))
        _swiglu_kernel_autotuned[grid](
            x, y, output,
            x.stride(0), y.stride(0), output.stride(0),
            N,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (M, triton.cdiv(N, cfg["BLOCK_SIZE"]))
        _swiglu_kernel[grid](
            x, y, output,
            x.stride(0), y.stride(0), output.stride(0),
            N,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return output


def get_last_config() -> dict | None:
    cfg = getattr(_swiglu_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
