import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 256, "threads": 128}
_last_autotune_config: dict = {}


def gaussian_blur_configs():
    block_sizes = [256, 512, 1024, 2048]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in block_sizes
        for nt in threads
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
    BLOCK_SIZE: int = 256,
    threads: int = 128,
):
    total_elements = T.const("total_elements")
    input: T.Tensor((total_elements,), dtype)
    kernel: T.Tensor((kernel_rows * kernel_cols,), dtype)
    output: T.Tensor((total_elements,), dtype)

    center_r = kernel_rows // 2
    center_c = kernel_cols // 2

    with T.Kernel(T.ceildiv(total_elements, BLOCK_SIZE), threads=threads) as pid:
        start = pid * BLOCK_SIZE
        full_blocks = total_elements // BLOCK_SIZE
        tile_end = start + BLOCK_SIZE - 1
        row_first = start // input_cols
        row_last = tile_end // input_cols
        col_first = start % input_cols
        col_last = tile_end % input_cols
        kernel_local = T.alloc_fragment((kernel_rows * kernel_cols,), dtype)
        acc = T.alloc_fragment((BLOCK_SIZE,), "float32")
        T.copy(kernel[0], kernel_local)
        T.fill(acc, 0.0)

        if (
            pid < full_blocks
            and row_first == row_last
            and row_first >= center_r
            and row_last < input_rows - center_r
            and col_first >= center_c
            and col_last < input_cols - center_c
        ):
            for kr in T.unroll(kernel_rows):
                for kc in T.unroll(kernel_cols):
                    for i in T.Parallel(BLOCK_SIZE):
                        offset = start + i
                        row = offset // input_cols
                        col = offset % input_cols
                        input_idx = (row + kr - center_r) * input_cols + col + kc - center_c
                        kernel_idx = kr * kernel_cols + kc
                        acc[i] += (
                            T.Cast("float32", input[input_idx])
                            * T.Cast("float32", kernel_local[kernel_idx])
                        )

            T.copy(acc, output[start: start + BLOCK_SIZE])
        else:
            for kr in T.unroll(kernel_rows):
                for kc in T.unroll(kernel_cols):
                    for i in T.Parallel(BLOCK_SIZE):
                        offset = start + i
                        row = offset // input_cols
                        col = offset % input_cols
                        in_r = row + kr - center_r
                        in_c = col + kc - center_c
                        if (
                            offset < total_elements
                            and in_r >= 0
                            and in_r < input_rows
                            and in_c >= 0
                            and in_c < input_cols
                        ):
                            input_idx = in_r * input_cols + in_c
                            kernel_idx = kr * kernel_cols + kc
                            acc[i] += (
                                T.Cast("float32", input[input_idx])
                                * T.Cast("float32", kernel_local[kernel_idx])
                            )

            if pid < full_blocks:
                    T.copy(acc, output[start: start + BLOCK_SIZE])
            else:
                for i in T.Parallel(BLOCK_SIZE):
                    offset = start + i
                    if offset < total_elements:
                        output[offset] = T.Cast(dtype, acc[i])


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
    dtype = str(input.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(input, kernel, output):
            tuned_kernel = gaussian_blur_kernel.compile(
                input, kernel, output,
                input_rows=input_rows,
                input_cols=input_cols,
                kernel_rows=kernel_rows,
                kernel_cols=kernel_cols,
                dtype=dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input, kernel, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        gaussian_blur_kernel(
            input, kernel, output,
            input_rows=input_rows,
            input_cols=input_cols,
            kernel_rows=kernel_rows,
            kernel_cols=kernel_cols,
            dtype=dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
