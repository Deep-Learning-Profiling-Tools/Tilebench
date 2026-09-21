import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "threads": 128}
_last_autotune_config: dict = {}


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
        start = pid * BLOCK_SIZE
        x_reg = T.alloc_fragment((BLOCK_SIZE,), "float32")
        gate_reg = T.alloc_fragment((BLOCK_SIZE,), "float32")
        bias_reg = T.alloc_fragment((BLOCK_SIZE,), "float32")
        output_reg = T.alloc_fragment((BLOCK_SIZE,), "float32")
        T.copy(x[start : start + BLOCK_SIZE], x_reg)
        T.copy(gate[start : start + BLOCK_SIZE], gate_reg)
        T.copy(bias[start : start + BLOCK_SIZE], bias_reg)
        for local_idx in T.Parallel(BLOCK_SIZE):
            z = x_reg[local_idx] * gate_reg[local_idx] + bias_reg[local_idx]
            output_reg[local_idx] = z * T.sigmoid(z)
        T.copy(output_reg, output[start : start + BLOCK_SIZE])


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, autotune: bool = False, **kwargs) -> torch.Tensor:

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
        _last_autotune_config.clear()
        _last_autotune_config.update(dict(kernel.config or {}))
        kernel(x, gate, bias, output)
    else:
        _last_autotune_config.clear()
        cfg = _DEFAULT_CONFIG
        fused_activation_kernel(
            x, gate, bias, output, in_dtype=in_dtype, out_dtype=out_dtype,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            threads=cfg["threads"],
        )

    return output


def get_last_config() -> dict | None:
    return dict(_last_autotune_config) if _last_autotune_config else None
