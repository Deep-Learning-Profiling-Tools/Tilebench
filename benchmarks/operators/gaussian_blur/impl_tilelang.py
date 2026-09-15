import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_R": 8, "BLOCK_C": 64, "threads": 256}
_last_autotune_config: dict = {}


def gaussian_blur_configs():
    return [
        dict(BLOCK_R=br, BLOCK_C=bc, threads=nt)
        for br, bc in [(1, 128), (1, 256), (1, 512), (2, 128), (2, 256),
                       (4, 128), (4, 256), (8, 64)]
        for nt in [128, 256]
    ]


@tilelang.autotune(configs=gaussian_blur_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def gaussian_blur_kernel(
    input,
    kernel,
    output,
    input_rows,
    input_cols,
    kernel_rows,
    kernel_cols,
    dtype,
    BLOCK_R: int = 8,
    BLOCK_C: int = 64,
    threads: int = 256,
):
    input: T.Tensor((input_rows, input_cols), dtype)
    kernel: T.Tensor((kernel_rows * kernel_cols,), dtype)
    output: T.Tensor((input_rows, input_cols), dtype)

    center_r = kernel_rows // 2
    center_c = kernel_cols // 2

    with T.Kernel(
        T.ceildiv(input_rows, BLOCK_R),
        T.ceildiv(input_cols, BLOCK_C),
        threads=threads,
    ) as (pid_r, pid_c):
        acc = T.alloc_fragment((BLOCK_R, BLOCK_C), "float32")
        T.annotate_safe_value({input: 0.0})
        T.fill(acc, 0.0)

        row_start = pid_r * BLOCK_R
        col_start = pid_c * BLOCK_C
        for kr in T.unroll(kernel_rows):
            for kc in T.unroll(kernel_cols):
                w = T.cast(kernel[kr * kernel_cols + kc], "float32")
                for i, j in T.Parallel(BLOCK_R, BLOCK_C):
                    row = row_start + i
                    col = col_start + j
                    in_r = row + kr - center_r
                    in_c = col + kc - center_c
                    acc[i, j] += T.cast(input[in_r, in_c], "float32") * w

        for i, j in T.Parallel(BLOCK_R, BLOCK_C):
            output[row_start + i, col_start + j] = T.cast(acc[i, j], dtype)


def run(
    input,
    kernel,
    input_rows,
    input_cols,
    kernel_rows,
    kernel_cols,
    block_size: int = 1024,
    autotune: bool = False,
    **kwargs,
):
    total_elements = input_rows * input_cols
    if total_elements <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_elements, dtype=input.dtype, device=input.device)
    input_2d = input.view(input_rows, input_cols)
    output_2d = output.view(input_rows, input_cols)
    dtype = str(input.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(input_2d, kernel, output_2d):
            tuned_kernel = gaussian_blur_kernel.compile(
                input_2d, kernel, output_2d,
                input_rows=input_rows,
                input_cols=input_cols,
                kernel_rows=kernel_rows,
                kernel_cols=kernel_cols,
                dtype=dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input_2d, kernel, output_2d)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        gaussian_blur_kernel(
            input_2d, kernel, output_2d,
            input_rows=input_rows,
            input_cols=input_cols,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
            dtype=dtype,
            BLOCK_R=cfg["BLOCK_R"],
            BLOCK_C=cfg["BLOCK_C"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
