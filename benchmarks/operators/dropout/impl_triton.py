import torch
import triton
import triton.language as tl

from core.triton_tma import ensure_tma_available

_DEFAULT_CONFIG = {"BLOCK_SIZE": 1024, "num_warps": 4}


@triton.jit
def _dropout_kernel(
    x_ptr,
    x_keep_ptr,
    output_ptr,
    n_elements,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    x_desc = tl.make_tensor_descriptor(x_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    keep_desc = tl.make_tensor_descriptor(x_keep_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    out_desc = tl.make_tensor_descriptor(output_ptr, shape=[n_elements, 1], strides=[1, 1], block_shape=[BLOCK_SIZE, 1])
    x = x_desc.load([block_start, 0])
    x_keep = keep_desc.load([block_start, 0])
    output = tl.where(x_keep.to(tl.int1), x / (1 - p), 0.0)
    out_desc.store([block_start, 0], output)


_dropout_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": bs}, num_warps=nw)
        for bs in [512, 1024, 2048]
        for nw in [2, 4, 8]
    ],
    key=["n_elements"],
)(_dropout_kernel)


def run(x: torch.Tensor, x_keep: torch.Tensor, p: float,
        block_size: int = 1024, autotune: bool = False) -> torch.Tensor:
    ensure_tma_available()
    x_flat = x.contiguous().view(-1)
    keep_flat = x_keep.contiguous().view(-1)
    output = torch.empty_like(x_flat)
    n_elements = x_flat.numel()
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _dropout_kernel_autotuned[grid](x_flat, keep_flat, output, n_elements, p)
    else:
        cfg = _DEFAULT_CONFIG
        grid = (triton.cdiv(n_elements, cfg["BLOCK_SIZE"]),)
        _dropout_kernel[grid](
            x_flat, keep_flat, output, n_elements, p,
            BLOCK_SIZE=cfg["BLOCK_SIZE"],
            num_warps=cfg["num_warps"],
        )
    return output.view(x.shape)


def get_last_config() -> dict | None:
    cfg = getattr(_dropout_kernel_autotuned, "best_config", None)
    if cfg is None:
        return None
    return {"BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"], "num_warps": cfg.num_warps}
