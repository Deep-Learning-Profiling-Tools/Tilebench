import torch
import triton
import triton.language as tl

_last_config: dict | None = None


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [256, 512, 1024, 2048, 4096, 8192]
        for nw in [4, 8, 16]
    ],
    key=["n_elements"],
)
@triton.jit
def mul2_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2
    tl.store(output_ptr + offsets, y, mask=mask)


def run(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    global _last_config
    n_elements = x.numel()
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    mul2_kernel[grid](x, output, n_elements)
    cfg = mul2_kernel.best_config
    if cfg is not None:
        _last_config = {
            "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
            "num_warps":  cfg.num_warps,
        }
    return output


def get_last_config() -> dict | None:
    return _last_config
