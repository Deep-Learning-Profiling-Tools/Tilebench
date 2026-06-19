import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"threads": 128}
_last_autotune_config: dict = {}


def dequantize_rowwise_configs():
    threads = [64, 128, 256, 512]
    return [
        dict(threads=nt)
        for nt in threads
    ]


@tilelang.autotune(configs=dequantize_rowwise_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def dequantize_rowwise_kernel(x, state_x, output, in_dtype, state_dtype, out_dtype, threads: int = 128):
    rows = T.dynamic("rows")
    cols = T.const("cols")
    x: T.Tensor((rows, cols), in_dtype)
    state_x: T.Tensor((rows,), state_dtype)
    output: T.Tensor((rows, cols), out_dtype)

    with T.Kernel(rows, threads=threads) as row:
        scale = state_x[row] * (1.0 / 127.0)
        for col in T.Parallel(cols):
            value = T.cast(x[row, col], "float32") * scale
            output[row, col] = T.cast(value, out_dtype)


def run(x: torch.Tensor, state_x: torch.Tensor, autotune: bool = False, **kwargs) -> torch.Tensor:
    in_dtype = str(x.dtype).removeprefix("torch.")
    state_dtype = str(state_x.dtype).removeprefix("torch.")
    output = torch.empty(x.shape, device=x.device, dtype=torch.float16)
    out_dtype = str(output.dtype).removeprefix("torch.")

    if autotune:
        with set_autotune_inputs(x, state_x, output):
            kernel = dequantize_rowwise_kernel.compile(
                x, state_x, output,
                in_dtype=in_dtype,
                state_dtype=state_dtype,
                out_dtype=out_dtype,
            )
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, state_x, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        dequantize_rowwise_kernel(
            x, state_x, output,
            in_dtype=in_dtype,
            state_dtype=state_dtype,
            out_dtype=out_dtype,
            threads=cfg["threads"],
        )
    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
