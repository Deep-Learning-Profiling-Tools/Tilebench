import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def conv2d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    input_cols, output_cols, total_out,
    kernel_rows: tl.constexpr, kernel_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-channel VALID 2D correlation — direct stencil, 1D-flat tiling
    (matches 1d_conv / 3d_conv and impl_cutile.py). Each program owns
    BLOCK_SIZE consecutive flattened output positions, decodes them to
    (oh, ow), and accumulates kernel[i,j] * input[oh+i, ow+j] over the window.

    output[oh, ow] = sum_{i,j} kernel[i,j] * input[oh+i, ow+j]
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    # Decode flat output offset -> (oh, ow)
    oh = offsets // output_cols
    ow = offsets % output_cols

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(kernel_rows):
        for j in tl.static_range(kernel_cols):
            # input[oh+i, ow+j] as a 1D flat index (contiguous: row stride = input_cols)
            input_idx = (oh + i) * input_cols + (ow + j)
            x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
            w = tl.load(kernel_ptr + i * kernel_cols + j)
            acc += x * w

    tl.store(output_ptr + offsets, acc, mask=mask)


_conv2d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [256, 512, 1024, 2048]
        for nw in [2, 4, 8]
        for ns in [1, 2, 3]
    ],
    key=["input_cols", "output_cols", "total_out"],
    warmup=1,
    rep=3,
)(conv2d_kernel)


def run(input: torch.Tensor, kernel: torch.Tensor,
        input_rows: int, input_cols: int,
        kernel_rows: int, kernel_cols: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    """
    Triton single-channel VALID 2D correlation — direct stencil (1D-flat tiling).

    input:  [input_rows, input_cols]
    kernel: [kernel_rows, kernel_cols]
    output: [input_rows - kernel_rows + 1, input_cols - kernel_cols + 1]
    """
    assert input.is_contiguous() and kernel.is_contiguous()
    out_rows = input_rows - kernel_rows + 1
    out_cols = input_cols - kernel_cols + 1
    total_out = out_rows * out_cols
    output = torch.empty((out_rows, out_cols), device=input.device, dtype=input.dtype)

    # Contiguous 2D tensors index correctly via base_ptr + flat element offset.
    kw = dict(
        input_ptr=input, kernel_ptr=kernel, output_ptr=output,
        input_cols=input_cols, output_cols=out_cols, total_out=total_out,
        kernel_rows=kernel_rows, kernel_cols=kernel_cols,
    )

    if autotune:
        grid = lambda meta: (triton.cdiv(total_out, meta["BLOCK_SIZE"]),)
        _conv2d_kernel_autotuned[grid](**kw)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(total_out, cfg["BLOCK_SIZE"]),)
        conv2d_kernel[grid](
            **kw,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_conv2d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
        "num_stages": cfg.num_stages,
    }
