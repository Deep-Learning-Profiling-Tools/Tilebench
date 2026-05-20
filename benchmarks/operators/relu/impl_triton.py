import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4, "num_stages": 2}


@triton.jit
def _relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    out_desc = tl.make_tensor_descriptor(output_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    x = x_desc.load([block_start, 0])
    y = tl.where(x >= 0, x, tl.zeros_like(x))
    out_desc.store([block_start, 0], y)


_relu_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["n_elements"],
)(_relu_kernel)


def run(x: torch.Tensor, block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    ensure_tma_available()
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    output = torch.empty_like(x_flat)
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _relu_kernel_autotuned[grid](x_flat, output, n_elements)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _relu_kernel[grid](
            x_flat, output, n_elements,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
        )
    return output.view(x.shape)


def get_last_config() -> dict | None:
    cfg = getattr(_relu_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
