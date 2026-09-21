import torch 
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs
_DEFAULT_CONFIG = {"BLOCK_SIZE": 2048, "threads": 128}
_last_autotune_config: dict = {}
def quantize_global_configs():
    BLOCK_SIZE = [2048, 4096, 8192, 16384]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE = bs, threads = nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs = quantize_global_configs(), warmup = 20, rep = 100, timeout = 60)
@tilelang.jit


def quantize_global_kernel(x, output, in_dtype, out_dtype, BLOCK_SIZE: int = 2048, threads: int = 128):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements, ), in_dtype)
    output: T.Tensor((n_elements, ), out_dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads = threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            output[idx] = T.cast(x[idx], "float16")

def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    in_dtype = str(x.dtype).removeprefix("torch.")
    output = torch.empty(x.shape, device = x.device, dtype = torch.float16)
    out_dtype = str(output.dtype).removeprefix("torch.")
    if autotune:
        with set_autotune_inputs(x, output):
            kernel = quantize_global_kernel.compile(x, output, 
                                                     in_dtype = in_dtype, 
                                                     out_dtype = out_dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        quantize_global_kernel(
            x, output, in_dtype = in_dtype, out_dtype = out_dtype,
            BLOCK_SIZE = cfg["BLOCK_SIZE"],
            threads = cfg["threads"],
        )
    return output
def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
