import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 256, "num_warps": 4}


@triton.jit
def gaussian_blur_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_rows,
    input_cols,
    total_elements,
    kernel_rows: tl.constexpr,
    kernel_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    row = offsets // input_cols
    col = offsets % input_cols

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    center_r = kernel_rows // 2
    center_c = kernel_cols // 2

    for kr in tl.static_range(0, kernel_rows):
        for kc in tl.static_range(0, kernel_cols):
            in_r = row + kr - center_r
            in_c = col + kc - center_c

            valid = (
                mask
                & (in_r >= 0)
                & (in_r < input_rows)
                & (in_c >= 0)
                & (in_c < input_cols)
            )

            input_idx = in_r * input_cols + in_c
            kernel_idx = kr * kernel_cols + kc

            x = tl.load(input_ptr + input_idx, mask=valid, other=0.0)
            w = tl.load(kernel_ptr + kernel_idx)

            acc += x * w

    tl.store(output_ptr + offsets, acc, mask=mask)


_gaussian_blur_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048]
        for nw in [4, 8]
    ],
    key=["total_elements"],
)(gaussian_blur_kernel)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    total_elements = input_rows * input_cols
    if total_elements <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_elements, dtype=input.dtype, device=input.device)

    if autotune:
        grid = lambda meta: (triton.cdiv(total_elements, meta["BLOCK_SIZE"]),)
        _gaussian_blur_kernel_autotuned[grid](
            input, kernel, output,
            input_rows, input_cols, total_elements,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(total_elements, cfg["BLOCK_SIZE"]),)
        gaussian_blur_kernel[grid](
            input, kernel, output,
            input_rows, input_cols, total_elements,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_gaussian_blur_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
        "num_warps": cfg.num_warps,
    }
