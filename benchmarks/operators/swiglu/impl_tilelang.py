import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def swiglu_configs():
    BLOCK_SIZE = [512, 1024, 2048, 4096, 8192, 16384]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs=swiglu_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def swiglu_kernel(x, y, output, dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements,), dtype)
    y: T.Tensor((n_elements,), dtype)
    output: T.Tensor((n_elements,), dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads=threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            x_f32 = T.cast(x[idx], "float32")
            y_f32 = T.cast(y[idx], "float32")
            output[idx] = T.cast(x_f32 * T.sigmoid(x_f32) * y_f32, dtype)


def run(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:

    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape.")

    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    output = torch.empty_like(x_flat)
    dtype = str(x.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(x_flat, y_flat, output):
            kernel = swiglu_kernel.compile(x_flat, y_flat, output, dtype=dtype)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x_flat, y_flat, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        swiglu_kernel(
            x_flat, y_flat, output, dtype=dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output.view(x.shape)


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
