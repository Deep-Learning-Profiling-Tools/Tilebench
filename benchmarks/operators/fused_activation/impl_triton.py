"""Triton fused element-wise activation: out = silu(x * gate + bias).

silu(z) = z * sigmoid(z). 1D grid; each program handles BLOCK_SIZE
elements. Pure bandwidth-bound (3 reads + 1 write per element +
sigmoid via SFU).
"""
import torch
import triton
import triton.language as tl


_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}


@triton.jit
def _fused_activation_kernel(
    x_ptr, gate_ptr, bias_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x * gate + bias
    out = z * tl.sigmoid(z)  # SiLU
    tl.store(out_ptr + offsets, out, mask=mask)


_fused_activation_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048, 4096, 8192]
        for nw in [4, 8, 16]
    ],
    key=["n_elements"],
)(_fused_activation_kernel)


def run(x: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    if x.shape != gate.shape or x.shape != bias.shape:
        raise ValueError("All input tensors must have the same shape.")
    x = x.contiguous()
    gate = gate.contiguous()
    bias = bias.contiguous()
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
    return out


def get_last_config() -> dict | None:
    cfg = getattr(_fused_activation_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
