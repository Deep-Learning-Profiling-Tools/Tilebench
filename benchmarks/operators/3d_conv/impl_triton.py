import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 256, "num_warps": 4}


@triton.jit
def _conv3d_kernel(
    input_ptr, kernel_ptr, output_ptr,
    input_depth, input_rows, input_cols,
    output_depth, output_rows, output_cols,
    total_out,
    kernel_depth: tl.constexpr,
    kernel_rows: tl.constexpr,
    kernel_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    output_plane = output_rows * output_cols
    output_depth_id = offsets // output_plane
    output_remain = offsets % output_plane
    output_row_id = output_remain // output_cols
    output_col_id = output_remain % output_cols

    input_plane = input_rows * input_cols
    kernel_plane = kernel_rows * kernel_cols

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kd in tl.static_range(kernel_depth):
        for kr in tl.static_range(kernel_rows):
            for kc in tl.static_range(kernel_cols):
                input_idx = (output_depth_id + kd) * input_plane + (output_row_id + kr) * input_cols + (output_col_id + kc)
                kernel_idx = kd * kernel_plane + kr * kernel_cols + kc
                x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
                w = tl.load(kernel_ptr + kernel_idx)
                acc += x * w

    tl.store(output_ptr + offsets, acc, mask=mask)


_conv3d_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048]
        for nw in [2, 4, 8, 16]
    ],
    key=["total_out"],
)(_conv3d_kernel)


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    total_out = output_depth * output_rows * output_cols

    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=torch.float32, device=input.device)

    if autotune:
        grid = lambda meta: (triton.cdiv(total_out, meta["BLOCK_SIZE"]),)
        _conv3d_kernel_autotuned[grid](
            input, kernel, output,
            input_depth, input_rows, input_cols,
            output_depth, output_rows, output_cols,
            total_out,
            kernel_depth=kernel_depth,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(total_out, cfg["BLOCK_SIZE"]),)
        _conv3d_kernel[grid](
            input, kernel, output,
            input_depth, input_rows, input_cols,
            output_depth, output_rows, output_cols,
            total_out,
            kernel_depth=kernel_depth,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )

    return output.to(input.dtype)


def get_last_config() -> dict | None:
    cfg = getattr(_conv3d_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
