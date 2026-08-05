import torch 
import tilelang 
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 256, "threads": 128}
_last_autotune_config: dict = {}
def conv3d_configs():
    BLOCK_SIZE = [256, 512, 1024, 2048]
    threads = [64, 128, 256, 512]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]

@tilelang.autotune(configs=conv3d_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def conv3d_kernel(
    input, kernel, output, output_rows, output_cols,
    input_rows, input_cols, 
    kernel_depth, kernel_rows, kernel_cols,
    dtype,
    BLOCK_SIZE: int = 256,
    threads: int = 128,
):
    in_elements = T.const("in_elements")
    out_elements = T.const("out_elements")
    k_elements = T.const("k_elements")
    
    input: T.Tensor((in_elements, ), dtype)
    output: T.Tensor((out_elements, ), "float32")
    kernel: T.Tensor((k_elements, ), dtype)
    with T.Kernel(T.ceildiv(out_elements, BLOCK_SIZE), threads=threads) as pid:
        start = pid * BLOCK_SIZE
        # CTA is for output
        # no padding 
        output_plane = output_rows * output_cols
        output_depth_id = T.alloc_fragment((BLOCK_SIZE, ), "int32")
        output_remain = T.alloc_fragment((BLOCK_SIZE, ), "int32")
        output_row_id = T.alloc_fragment((BLOCK_SIZE), "int32")
        output_col_id = T.alloc_fragment((BLOCK_SIZE), "int32")
        input_plane = input_rows * input_cols

        for i in T.Parallel(BLOCK_SIZE):
            output_depth_id[i] = (start + i) // (output_plane)
            output_remain[i] = (start + i) % output_plane
            output_row_id[i] = (output_remain[i]) // output_cols
            output_col_id[i] = (output_remain[i]) % output_cols

        acc = T.alloc_fragment((BLOCK_SIZE, ), "float32")
        T.fill(acc, 0.0)
        #x = T.alloc_fragment((BLOCK_SIZE,), dtype)

        for kd in T.serial(kernel_depth):
            for kr in T.serial(kernel_rows):
                for kc in T.serial(kernel_cols):
                    
                    for i in T.Parallel(BLOCK_SIZE):
                        in_idx = ((output_depth_id[i] + kd) * input_plane 
                        + (output_row_id[i] + kr) * input_cols + (output_col_id[i] + kc))
                        w_idx = kd * (kernel_rows * kernel_cols) + kr * kernel_cols + kc
                        acc[i] += input[in_idx] * kernel[w_idx]
        T.copy(acc, output[start: start + BLOCK_SIZE])


def run(input, kernel, input_depth, input_rows, input_cols,
        kernel_depth, kernel_rows, kernel_cols,
        block_size: int = 1024, autotune: bool = False, **kwargs):
    output_depth = input_depth - kernel_depth + 1
    output_rows = input_rows - kernel_rows + 1
    output_cols = input_cols - kernel_cols + 1
    total_out = output_depth * output_rows * output_cols
    dtype = str(input.dtype).removeprefix("torch.")
    if total_out <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(total_out, dtype=torch.float32, device=input.device)

    if autotune:
        with set_autotune_inputs(input, kernel, output):
            tuned_kernel = conv3d_kernel.compile(
                input, kernel, output, output_rows=output_rows,
                output_cols=output_cols, input_rows = input_rows,
                kernel_depth=kernel_depth, kernel_rows=kernel_rows,
                kernel_cols=kernel_cols, input_cols=input_cols, dtype=dtype
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input, kernel, output)

    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        conv3d_kernel(
                input, kernel, output, output_rows=output_rows,
                output_cols=output_cols, input_rows = input_rows, kernel_depth=kernel_depth, kernel_rows=kernel_rows,
                kernel_cols=kernel_cols, input_cols=input_cols, dtype=dtype, BLOCK_SIZE=cfg["BLOCK_SIZE"],
                threads=cfg["threads"]
            )

    return output.to(input.dtype)

def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
            
                         


    
