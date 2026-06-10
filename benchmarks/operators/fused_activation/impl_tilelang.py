import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config = None


def fused_activation_configs():
    BLOCK_SIZE = [512, 1024, 2048]
    threads = [64, 128, 256]
    return [
        dict(BLOCK_SIZE=bs, threads=nt)
        for bs in BLOCK_SIZE
        for nt in threads
    ]


@tilelang.autotune(configs=fused_activation_configs(), warmup=20, rep=100, timeout=60)
@tilelang.jit
def fused_activation_kernel(x, gate, bias, output, in_dtype, out_dtype, BLOCK_SIZE: int = 1024, threads: int = 128):
    n_elements = T.const("n_elements")
    x: T.Tensor((n_elements,), in_dtype)
    gate: T.Tensor((n_elements,), in_dtype)
    bias: T.Tensor((n_elements,), in_dtype)
    output: T.Tensor((n_elements,), out_dtype)

    with T.Kernel(T.ceildiv(n_elements, BLOCK_SIZE), threads=threads) as pid:
        for local_idx in T.Parallel(BLOCK_SIZE):
            idx = local_idx + pid * BLOCK_SIZE
            if idx < n_elements:
                #triton converts to fp32 so just copy that
                z = T.cast(x[idx], "float32") * T.cast(gate[idx], "float32") + T.cast(bias[idx], "float32")
                output[idx] = z * T.sigmoid(z)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, autotune: bool = False, **kwargs) -> torch.Tensor:
    global _last_autotune_config

    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")

    x = x.contiguous()
    gate = gate.contiguous()
    bias = bias.contiguous()
    in_dtype = str(x.dtype).removeprefix("torch.")
    out_dtype = "float32"
    output = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    if autotune:
        with set_autotune_inputs(x, gate, bias, output):
            kernel = fused_activation_kernel.compile(x, gate, bias, output, in_dtype=in_dtype, out_dtype=out_dtype)
        _last_autotune_config = dict(kernel.config or {})
        kernel(x, gate, bias, output)
    else:
        _last_autotune_config = None
        cfg = _DEFAULT_CONFIG
        fused_activation_kernel(
            x, gate, bias, output, in_dtype=in_dtype, out_dtype=out_dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return _last_autotune_config
