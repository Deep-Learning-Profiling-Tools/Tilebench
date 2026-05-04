import torch
import triton
import triton.language as tl

_DEFAULT_CONFIG = {"BLOCK_SIZE": 2048, "num_warps": 4, "num_stages": 2}


@triton.jit
def _quantize_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x.to(tl.float16), mask=mask)


_quantize_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw, num_stages=ns)
        for bs in [2048, 4096, 8192, 8192, 16384]
        for nw in [2, 4, 8]
        for ns in [1, 2, 4]
        if bs >= nw * 32 * 4
    ],
    key=["n_elements"],
)(_quantize_kernel)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty(x.shape, device=x.device, dtype=torch.float16)
    n_elements = x.numel()
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _quantize_kernel_autotuned[grid](x, out, n_elements)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _quantize_kernel[grid](
            x, out, n_elements,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )
    return out


def get_last_config() -> dict | None:
    cfg = getattr(_quantize_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
