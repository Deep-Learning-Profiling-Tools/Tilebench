import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def conv1d_configs():
    BLOCK_SIZE = [256, 512, 1024, 2048]
    threads = [64, 128, 256, 512]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]

@tilelang.autotune(configs=conv1d_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def conv1d_kernel(input, kernel, output, dtype, 
                  BLOCK_SIZE: int = 1024, 
                  threads: int = 128):
    N = T.const("N")
    kernel_size = T.const("kernel_size")
    output_size = N - kernel_size + 1
    input: T.Tensor((N,), dtype)
    kernel: T.Tensor((kernel_size, ), dtype)
    output: T.Tensor((N - kernel_size + 1, ), dtype)
    with T.Kernel(T.ceildiv(output_size, BLOCK_SIZE), threads=threads) as pid:
        acc = T.alloc_fragment((BLOCK_SIZE, ), "float32")
        T.fill(acc, 0.0)
        start = pid * BLOCK_SIZE
        full_blocks = output_size // BLOCK_SIZE

        if pid < full_blocks:
            for i in T.unroll(0, kernel_size):
                for j in T.Parallel(BLOCK_SIZE):
                    acc[j] += (
                        T.Cast("float32", input[start + j + i])
                        * T.Cast("float32", kernel[i])
                    )

            T.copy(acc, output[start: start + BLOCK_SIZE])
        else:
            for i in T.unroll(0, kernel_size):
                for j in T.Parallel(BLOCK_SIZE):
                    if start + j < output_size:
                        acc[j] += (
                            T.Cast("float32", input[start + j + i])
                            * T.Cast("float32", kernel[i])
                        )

            for j in T.Parallel(BLOCK_SIZE):
                if start + j < output_size:
                    output[start + j] = T.Cast(dtype, acc[j])

def run(input: torch.Tensor, kernel: torch.Tensor,
        input_size: int, kernel_size: int,
        block_size: int = 1024, autotune: bool = False, **kwargs):

    output_size = input_size - kernel_size + 1

    if output_size <= 0:
        return torch.empty(0, dtype=input.dtype, device=input.device)

    output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    dtype = str(input.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(input, kernel, output):
            tuned_kernel = conv1d_kernel.compile(
                input, kernel, output,
                dtype=dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(tuned_kernel.config or {}))
        tuned_kernel(input, kernel, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        conv1d_kernel(
            input, kernel, output, dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
