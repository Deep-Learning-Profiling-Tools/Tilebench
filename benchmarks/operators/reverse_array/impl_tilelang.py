import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config = None


def reverse_array_configs():
    BLOCK_SIZE = [1024, 2048, 4096, 8192]
    threads = [128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs = reverse_array_configs(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit
def reverse_array_kernel(input, output, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    input: T.Tensor((n_elements, ), dtype)
    output: T.Tensor((n_elements, ), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads = threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            if idx < n_elements:
                output[idx] = input[n_elements - 1 - idx]
                
def run(input: torch.Tensor, N : int, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:
    global _last_autotune_config
    dtype = str(input.dtype).removeprefix("torch.")
    output = torch.empty_like(input)
    if autotune:
        with set_autotune_inputs(input, output):
            kernel = reverse_array_kernel.compile(input, output, dtype = dtype)
        _last_autotune_config = dict(kernel.config)
        kernel(input, output)
    else:
        _last_autotune_config = None
        cfg = _DEFAULT_CONFIG
        reverse_array_kernel(
            input, output, dtype = dtype,
            BLOCK_SIZE = cfg["BLOCK_SIZE"],
            threads = cfg["threads"],
        )
    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
