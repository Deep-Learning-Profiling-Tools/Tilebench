"""2D Gaussian blur via 2D output tiles + shifted 2D loads: each program
owns a (BLOCK_R, BLOCK_C) output tile and accumulates kernel_rows *
kernel_cols shifted masked loads into an fp32 accumulator (49-tap
weighted sum — the fp32 accumulation is load-bearing, unlike selection
ops). No flat-offset decode (div/mod) per element."""
import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_R": 8, "BLOCK_C": 64, "num_warps": 8}


@triton.jit
def gaussian_blur_kernel(
    input_ptr,
    kernel_ptr,
    output_ptr,
    input_rows,
    input_cols,
    kernel_rows: tl.constexpr,
    kernel_cols: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)

    for kr in tl.static_range(0, kernel_rows):
        for kc in tl.static_range(0, kernel_cols):
            in_r = r + (kr - kernel_rows // 2)
            in_c = c + (kc - kernel_cols // 2)
            valid = ((in_r[:, None] >= 0) & (in_r[:, None] < input_rows)
                     & (in_c[None, :] >= 0) & (in_c[None, :] < input_cols))
            x = tl.load(input_ptr + in_r[:, None] * input_cols + in_c[None, :],
                        mask=valid, other=0.0)
            w = tl.load(kernel_ptr + kr * kernel_cols + kc)
            acc += x * w

    out_mask = (r[:, None] < input_rows) & (c[None, :] < input_cols)
    tl.store(output_ptr + r[:, None] * input_cols + c[None, :], acc,
             mask=out_mask)


_gaussian_blur_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_R": br, "BLOCK_C": bc}, num_warps=nw)
        for br, bc in [(1, 128), (1, 256), (1, 512), (2, 128), (2, 256),
                       (4, 128), (4, 256), (8, 64)]
        for nw in [4, 8]
    ],
    key=["input_rows", "input_cols", "kernel_rows", "kernel_cols"],
)(gaussian_blur_kernel)


def run(input, kernel, input_rows, input_cols,
        kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    total_elements = input_rows * input_cols
    if total_elements <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_elements, dtype=input.dtype, device=input.device)

    if autotune:
        grid = lambda meta: (
            triton.cdiv(input_rows, meta["BLOCK_R"]),
            triton.cdiv(input_cols, meta["BLOCK_C"]),
        )
        _gaussian_blur_kernel_autotuned[grid](
            input, kernel, output,
            input_rows, input_cols,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
        )
    else:
        cfg = _DEFAULT_CONFIG
        grid = (
            triton.cdiv(input_rows, cfg["BLOCK_R"]),
            triton.cdiv(input_cols, cfg["BLOCK_C"]),
        )
        gaussian_blur_kernel[grid](
            input, kernel, output,
            input_rows, input_cols,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
            BLOCK_R=cfg["BLOCK_R"],
            BLOCK_C=cfg["BLOCK_C"],
            num_warps=cfg["num_warps"],
        )

    return output


def get_last_config() -> dict | None:
    cfg = getattr(_gaussian_blur_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {
        "BLOCK_R": cfg.kwargs["BLOCK_R"],
        "BLOCK_C": cfg.kwargs["BLOCK_C"],
        "num_warps": cfg.num_warps,
    }
