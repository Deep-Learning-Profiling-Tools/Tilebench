import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


def dropout_configs():
    BLOCK_SIZE = [512, 1024, 2048]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs=dropout_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def dropout_kernel(x, x_keep, output, dtype, p: float, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements,), dtype)
    x_keep: T.Tensor((n_elements,), dtype)
    output: T.Tensor((n_elements,), dtype)
    # compute division before parallel as less expensive
    scale = 1.0 / (1.0 - p)
    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads=threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            if idx < n_elements:
                if x_keep[idx] != 0:
                    output[idx] = x[idx] * scale
                else:
                    output[idx] = 0


def run(x: torch.Tensor, x_keep: torch.Tensor, p: float, block_size: int = 1024, autotune: bool = False, **kwargs) -> torch.Tensor:

    dtype = str(x.dtype).removeprefix("torch.")
    output = torch.empty_like(x)
    if autotune:
        with set_autotune_inputs(x, x_keep, output):
            kernel = dropout_kernel.compile(x, x_keep, output, dtype=dtype, p=p)
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, x_keep, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        dropout_kernel(
            x, x_keep, output, dtype=dtype, p=p,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
