"""Triton fused element-wise activation: out = silu(x * gate + bias).

silu(z) = z * sigmoid(z). 1D grid; each program handles BLOCK_SIZE
elements. Pure bandwidth-bound (3 reads + 1 write per element +
sigmoid via SFU).
"""
import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available


_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}


@triton.jit
def _fused_activation_kernel(
    x_ptr, gate_ptr, bias_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    gate_desc = tl.make_tensor_descriptor(gate_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    bias_desc = tl.make_tensor_descriptor(bias_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    out_desc = tl.make_tensor_descriptor(out_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])

    x = x_desc.load([block_start, 0]).to(tl.float32)
    gate = gate_desc.load([block_start, 0]).to(tl.float32)
    bias = bias_desc.load([block_start, 0]).to(tl.float32)
    z = x * gate + bias
    out = z * tl.sigmoid(z)  # SiLU
    out_desc.store([block_start, 0], out)


_fused_activation_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["n_elements"],
)(_fused_activation_kernel)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    ensure_tma_available()
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")
    out_shape = x.shape
    x = x.contiguous().view(-1)
    gate = gate.contiguous().view(-1)
    bias = bias.contiguous().view(-1)
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n_elements = x.numel()

    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _fused_activation_kernel_autotuned[grid](x, gate, bias, out, n_elements)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _fused_activation_kernel[grid](
            x, gate, bias, out, n_elements,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )
    return out.view(out_shape)


def get_last_config() -> dict | None:
    cfg = getattr(_fused_activation_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
