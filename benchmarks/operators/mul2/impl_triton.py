import torch
import triton
import triton.language as tl

_last_config: dict | None = None

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def _mul2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(output_ptr + offsets, y, mask=mask)


_mul2_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048, 4096, 8192]
        for nw in [4, 8, 16]
    ],
    key=["n_elements"],
)(_mul2_kernel)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    global _last_config
    n_elements = x.numel()
    output = torch.empty_like(x)
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _mul2_kernel_autotuned[grid](x, output, n_elements)
        cfg = _mul2_kernel_autotuned.best_config
        if cfg is not None:
            _last_config = {
                "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
                "num_warps":  cfg.num_warps,
            }
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _mul2_kernel[grid](
            x, output, n_elements,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return output


def get_last_config() -> dict | None:
    return _last_config
